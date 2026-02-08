"""
Stratified train/validation splitting for YOLO-style segmentation datasets.

This module partitions a YOLO-formatted segmentation dataset into training
and validation subsets while preserving, as closely as possible, the empirical
class distribution observed in the full dataset.

The split operates at the image level: each image and its associated annotation
file are assigned either to the `train/` or `val/` subset. Class membership is
inferred directly from YOLO polygon annotations by parsing the first token
(class ID) of each label line.

Special care is taken to handle edge cases commonly encountered in real-world
datasets:
- Images without annotations are explicitly supported and assigned an empty
  label file to maintain YOLO directory consistency.
- Rare classes are prioritised during validation set construction to maximise
  class coverage.
- The procedure is deterministic given a fixed dataset structure and random
  seed, ensuring reproducibility.

The splitting strategy is heuristic but principled:
1. Estimate per-class targets for the validation set based on the desired
   validation fraction.
2. Iteratively assign images to validation, prioritising under-represented
   (rare) classes and minimally annotated images.
3. Fill any remaining validation slots by selecting images that best improve
   the match to per-class targets.

Once the split is completed, files are physically rearranged on disk into
their respective `train/` and `val/` directories, and a concise per-class
summary is logged for verification.
"""


import os
import random
import shutil
from glob import glob
from collections import Counter
from hfinder.core import log as HF_log
from hfinder.session import folders as HF_folders
from hfinder.session import settings as HF_settings


def _discover_training_images(img_dir: str) -> list[str]:
    """Return sorted list of training image paths (YOLO layout).

    Parameters
    ----------
    img_dir : str
        Directory containing training images.

    Returns
    -------
    list[str]
        Sorted list of absolute image paths (e.g. *.jpg).
    """
    return sorted(glob(os.path.join(img_dir, "*.jpg")))


def _extract_classes_per_image(img_paths: list[str], lbl_dir: str) -> tuple[list[set[int]], set[int]]:
    """Parse YOLO label files and infer class presence per image.

    For each image path, the corresponding label file is assumed to be
    `<stem>.txt` in `lbl_dir`. Each non-empty line is expected to start with a
    class ID (integer). Malformed lines are ignored.

    Parameters
    ----------
    img_paths : list[str]
        List of image paths.
    lbl_dir : str
        Directory containing YOLO label files.

    Returns
    -------
    (list[set[int]], set[int])
        - y_per_img: list where each element is the set of class IDs present
          in the image.
        - all_classes: union of all class IDs observed across images.
    """
    y_per_img: list[set[int]] = []
    all_classes: set[int] = set()

    for ip in img_paths:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, stem + ".txt")

        cls_set: set[int] = set()
        if os.path.isfile(lp):
            with open(lp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tok0 = line.split()[0]
                    try:
                        cls_set.add(int(tok0))
                    except Exception:
                        continue

        y_per_img.append(cls_set)
        all_classes |= cls_set

    return y_per_img, all_classes


def _compute_validation_targets(y_per_img: list[set[int]], validation_frac: float) -> tuple[int, Counter, dict[int, float]]:
    """Compute overall and per-class soft targets for the validation set.

    The per-class targets are *soft* (floating-point), derived as:
        target_per_class_val[c] = validation_frac * total_per_class[c]

    Parameters
    ----------
    y_per_img : list[set[int]]
        Class sets per image.
    validation_frac : float
        Desired validation fraction (e.g. 0.2).

    Returns
    -------
    (int, Counter, dict[int, float])
        - target_val: desired number of validation images (>= 1).
        - total_per_class: counts of images containing each class.
        - target_per_class_val: soft per-class targets for validation.
    """
    n = len(y_per_img)
    target_val = max(1, int(round(validation_frac * n)))

    total_per_class = Counter()
    for s in y_per_img:
        for c in s:
            total_per_class[c] += 1

    target_per_class_val = {c: validation_frac * total_per_class[c] for c in total_per_class}
    return target_val, total_per_class, target_per_class_val


def _build_indices_per_class(y_per_img: list[set[int]], all_classes: set[int], rng: random.Random) -> dict[int, list[int]]:
    """Build and shuffle inverted indices: class -> image indices containing it."""
    indices_per_class = {c: [] for c in all_classes}
    for i, s in enumerate(y_per_img):
        for c in s:
            indices_per_class[c].append(i)
    for c in indices_per_class:
        rng.shuffle(indices_per_class[c])
    return indices_per_class


def _assign_validation_greedy(
    y_per_img: list[set[int]],
    all_classes: set[int],
    indices_per_class: dict[int, list[int]],
    target_val: int,
    target_per_class_val: dict[int, float],
) -> tuple[list[bool], Counter, int]:
    """Greedy validation assignment prioritising under-covered (often rare) classes.

    Strategy
    --------
    While budget remains:
      1) choose class with largest remaining unmet target;
      2) pick among its unassigned images the one with the fewest labels.

    Returns
    -------
    (in_val, cur_per_class_val, val_count)
    """
    n = len(y_per_img)
    assigned = [False] * n
    in_val = [False] * n
    cur_per_class_val = Counter()
    val_count = 0

    def remaining_need(c: int) -> float:
        return target_per_class_val.get(c, 0.0) - cur_per_class_val[c]

    # Work on a local copy: we may remove classes that become impossible to satisfy.
    mutable_classes = set(all_classes)

    while val_count < target_val:
        candidates = [c for c in mutable_classes if remaining_need(c) > 1e-9]
        if not candidates:
            break

        c_star = max(candidates, key=remaining_need)
        pool = [i for i in indices_per_class.get(c_star, []) if not assigned[i]]
        if not pool:
            mutable_classes.remove(c_star)
            continue

        i_star = min(pool, key=lambda i: len(y_per_img[i]) if y_per_img[i] else 0)

        assigned[i_star] = True
        in_val[i_star] = True
        val_count += 1
        for c in y_per_img[i_star]:
            cur_per_class_val[c] += 1

    # Mark remaining unassigned for downstream fill/train
    return in_val, cur_per_class_val, val_count, assigned


def _fill_validation_soft(
    y_per_img: list[set[int]],
    in_val: list[bool],
    assigned: list[bool],
    cur_per_class_val: Counter,
    target_val: int,
    target_per_class_val: dict[int, float],
) -> None:
    """Fill remaining validation slots by maximising improvement toward soft targets."""

    def remaining_need(c: int) -> float:
        return target_per_class_val.get(c, 0.0) - cur_per_class_val[c]

    def gain_if_val(i: int) -> float:
        return sum(max(0.0, remaining_need(c)) for c in y_per_img[i])

    remaining = [i for i in range(len(y_per_img)) if not assigned[i]]
    remaining.sort(key=lambda i: (gain_if_val(i), -len(y_per_img[i])), reverse=True)

    val_count = sum(in_val)
    for i in remaining:
        if val_count >= target_val:
            break
        assigned[i] = True
        in_val[i] = True
        val_count += 1
        for c in y_per_img[i]:
            cur_per_class_val[c] += 1


def _move_split_files(img_paths: list[str], lbl_dir: str, img_val_dir: str, lbl_val_dir: str, in_val: list[bool]) -> tuple[int, int]:
    """Move validation files; keep training files in place; ensure empty labels exist.

    Returns
    -------
    (moved_val, moved_train)
    """
    moved_val = moved_train = 0

    for i, ip in enumerate(img_paths):
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, stem + ".txt")

        if in_val[i]:
            shutil.move(ip, os.path.join(img_val_dir, os.path.basename(ip)))
            if os.path.exists(lp):
                shutil.move(lp, os.path.join(lbl_val_dir, os.path.basename(lp)))
            else:
                open(os.path.join(lbl_val_dir, stem + ".txt"), "a").close()
            moved_val += 1
        else:
            if not os.path.exists(lp):
                open(lp, "a").close()
            moved_train += 1

    return moved_val, moved_train


def _count_instances(lbl_path: str) -> Counter:
    """Count per-class instance lines across all label files in a directory."""
    cnt = Counter()
    for p in glob(os.path.join(lbl_path, "*.txt")):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cnt[int(line.split()[0])] += 1
                except Exception:
                    pass
    return cnt


def _strategy_preserve_distribution(
    *, y_per_img, target_val, rng,
    validation_frac, all_classes, total_per_class, indices_per_class, logger, **_
):
    """Default strategy: approximate preservation of empirical class distribution.

    Two-phase heuristic:
      (A) greedily satisfy unmet per-class validation targets (rare classes first),
      (B) fill remaining slots by maximising improvement toward soft targets.
    """
    target_per_class_val = {c: validation_frac * total_per_class[c] for c in total_per_class}

    in_val, cur_per_class_val, val_count, assigned = _assign_validation_greedy(
        y_per_img=y_per_img,
        all_classes=all_classes,
        indices_per_class=indices_per_class,
        target_val=target_val,
        target_per_class_val=target_per_class_val,
    )

    _fill_validation_soft(
        y_per_img=y_per_img,
        in_val=in_val,
        assigned=assigned,
        cur_per_class_val=cur_per_class_val,
        target_val=target_val,
        target_per_class_val=target_per_class_val,
    )
    return in_val


def split_train_val(validation_frac=0.2, seed=42, strategy=None):
    """Split a YOLO segmentation dataset into train/val using a pluggable strategy.

    This public entry point orchestrates I/O (discovery, parsing, moving files,
    and reporting) while delegating the *assignment policy* to a strategy
    function. This makes it straightforward to swap heuristics, e.g.:
      - preserve empirical distribution (current behaviour),
      - actively compensate class imbalance,
      - enforce minimum class coverage in validation,
      - optimise a custom objective.

    Parameters
    ----------
    validation_frac : float, default=0.2
        Fraction of images to allocate to the validation subset.
    seed : int, default=42
        Seed used for deterministic shuffles inside strategies.
    strategy : callable | None
        Assignment strategy with signature:

            strategy(y_per_img, target_val, rng, **context) -> list[bool]

        returning `in_val` (True for validation images). If None, the default
        distribution-preserving heuristic is used.

    Notes
    -----
    Stratification operates on *class presence per image* (multi-label).
    Reporting at the end logs per-class instance-line counts for convenience.
    """
    rng = random.Random(seed)

    img_dir = HF_folders.get_image_train_dir()
    lbl_dir = HF_folders.get_label_train_dir()
    img_val_dir = HF_folders.get_image_val_dir()
    lbl_val_dir = HF_folders.get_label_val_dir()

    # 1) Discover images
    img_paths = _discover_training_images(img_dir)
    if not img_paths:
        HF_log.warn("No training images found to split")
        return

    # 2) Parse labels → per-image class sets
    y_per_img, all_classes = _extract_classes_per_image(img_paths, lbl_dir)

    # 3) Compute validation budget (absolute number of images)
    target_val = max(1, int(round(validation_frac * len(img_paths))))

    # 4) Context: useful information a strategy may want to use
    #    (e.g., class frequencies, class names, inverted indices, etc.)
    total_per_class = Counter()
    for s in y_per_img:
        for c in s:
            total_per_class[c] += 1

    classes = HF_settings.load_class_list()
    indices_per_class = _build_indices_per_class(y_per_img, all_classes, rng)

    context = dict(
        validation_frac=validation_frac,
        all_classes=all_classes,
        total_per_class=total_per_class,
        class_names=classes,
        indices_per_class=indices_per_class,
        logger=HF_log,
    )

    # 5) Choose strategy (default: your current distribution-preserving heuristic)
    if strategy is None:
        strategy = _strategy_preserve_distribution  # defined below / elsewhere

    # 6) Delegate assignment
    in_val = strategy(y_per_img=y_per_img, target_val=target_val, rng=rng, **context)

    # Defensive checks: ensure correct length, boolean-like values, and exact budget
    if len(in_val) != len(img_paths):
        raise ValueError(f"strategy returned {len(in_val)} flags for {len(img_paths)} images")

    # Enforce exact validation size (strategy *should* already do it; this is a guardrail)
    val_idx = [i for i, b in enumerate(in_val) if b]
    if len(val_idx) != target_val:
        # Minimal coercion: keep the first target_val validation samples deterministically
        # (or you may prefer to raise an error instead)
        desired = set(val_idx[:target_val])
        in_val = [i in desired for i in range(len(in_val))]

    # 7) Move files and ensure empty labels exist
    moved_val, moved_train = _move_split_files(
        img_paths=img_paths,
        lbl_dir=lbl_dir,
        img_val_dir=img_val_dir,
        lbl_val_dir=lbl_val_dir,
        in_val=in_val,
    )

    # 8) Report counts
    val_counts = _count_instances(lbl_val_dir)
    train_counts = _count_instances(lbl_dir)

    HF_log.info(f"Split done. val images={moved_val}, train images={moved_train}")
    HF_log.info(f"Per-class (train): {dict(train_counts)}")
    HF_log.info(f"Per-class (val)  : {dict(val_counts)}")
