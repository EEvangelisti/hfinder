"""
Dataset splitting utilities for YOLO-based training.

This module performs a deterministic and stratified partitioning of a
YOLO-formatted dataset into training and validation subsets. Images are
associated with their corresponding annotation files, and classes are
inferred directly from the YOLO polygon labels. The procedure preserves
class diversity as much as possible: rare classes are allocated first,
followed by a balanced distribution of the remaining samples.

Main features:
    • Automatic detection of classes present in the dataset.
    • Stratified allocation based on class frequency.
    • Creation of empty annotation files when needed, ensuring YOLO
      directory integrity.
    • File organisation into `train/` and `val/` subdirectories.

The module exposes a single high-level entry point, `split_train_val`,
which can be invoked after all images and labels have been generated.
"""

import os
import random
import shutil
from glob import glob
from collections import Counter
from hfinder.core import log as HF_log
from hfinder.session import folders as HF_folders
from hfinder.session import settings as HF_settings


def split_train_val(validation_frac=0.2, seed=42):

    rng = random.Random(seed)

    img_dir      = HF_folders.get_image_train_dir()
    lbl_dir      = HF_folders.get_label_train_dir()
    img_val_dir  = HF_folders.get_image_val_dir()
    lbl_val_dir  = HF_folders.get_label_val_dir()

    img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    if not img_paths:
        HF_log.warn("No training images found to split")
        return

    y_per_img = []          # liste de sets de classes (ex: {0, 2})
    all_classes = set()
    for ip in img_paths:
        name = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, name + ".txt")
        cls_set = set()
        if os.path.isfile(lp):
            with open(lp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tok0 = line.split()[0]
                    try:
                        cls_id = int(tok0)
                        cls_set.add(cls_id)
                    except Exception:
                        continue
        y_per_img.append(cls_set)
        all_classes |= cls_set

    n = len(img_paths)
    target_val = max(1, int(round(validation_frac * n)))

    total_per_class = Counter()
    for s in y_per_img:
        for c in s: total_per_class[c] += 1

    total_all = sum(total_per_class.values())
    class_defs = HF_settings.load_class_definitions(keys='id')
    for c, count in sorted(total_per_class.items()):
        name = class_defs[c]
        pct = 100.0 * count / total_all if total_all else 0
        HF_log.info(f"{name:20s} ({c:2d}) : {count:6d} ({pct:5.2f}%)")

    target_per_class_val = {c: validation_frac * total_per_class[c] for c in total_per_class}

    # 3) Structures pour l'itérative stratification
    #    - pour chaque classe → indices d’images la contenant
    indices_per_class = {c: [] for c in all_classes}
    for i, s in enumerate(y_per_img):
        for c in s:
            indices_per_class[c].append(i)
    for c in indices_per_class:
        rng.shuffle(indices_per_class[c])

    assigned = [False] * n
    in_val   = [False] * n
    cur_per_class_val = Counter()
    val_count = 0

    # Fonction utilitaire: besoin restant par classe dans val
    def remaining_need(c):
        return target_per_class_val.get(c, 0.0) - cur_per_class_val[c]

    # 4) Étape A — satisfaire au mieux les classes rares d’abord
    #    Boucle tant qu’on peut améliorer et qu’il reste du budget val
    #    Heuristique: prendre la classe avec plus grand "need" restant,
    #    puis l’image la plus "rare" (moins d’étiquettes).
    while val_count < target_val:
        # Classe la plus "sous-représentée" côté val
        candidates_classes = [c for c in all_classes if remaining_need(c) > 1e-9]
        if not candidates_classes:
            break
        c_star = max(candidates_classes, key=lambda c: remaining_need(c))

        # Parmi ses images non assignées, prendre celle avec le moins de labels (rare)
        pool = [i for i in indices_per_class[c_star] if not assigned[i]]
        if not pool:
            # rien pour cette classe → on marquera comme impossible plus bas
            # On la “sature” pour éviter boucle infinie
            all_classes.remove(c_star)
            continue

        i_star = min(pool, key=lambda i: len(y_per_img[i]) if y_per_img[i] else 0)

        # Assigner à val
        assigned[i_star] = True
        in_val[i_star] = True
        val_count += 1
        for c in y_per_img[i_star]:
            cur_per_class_val[c] += 1

    # 5) Étape B — remplir le reste du budget val de façon douce (distance aux cibles)
    def gain_if_val(i):
        # somme des besoins pour les classes présentes dans i
        return sum(max(0.0, remaining_need(c)) for c in y_per_img[i])

    remaining = [i for i in range(n) if not assigned[i]]
    # Donner la priorité aux images qui améliorent le plus la couverture des classes
    remaining.sort(key=lambda i: (gain_if_val(i), -len(y_per_img[i])), reverse=True)

    for i in remaining:
        if val_count >= target_val:
            break
        # si l’image est négative, autoriser mais donner moins de priorité (déjà géré par tri)
        assigned[i] = True
        in_val[i] = True
        val_count += 1
        for c in y_per_img[i]:
            cur_per_class_val[c] += 1

    # 6) Le reste va en train
    for i in range(n):
        if not assigned[i]:
            in_val[i] = False

    # 7) Déplacement des fichiers
    moved_val, moved_train = 0, 0
    for i, ip in enumerate(img_paths):
        name = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, name + ".txt")
        dest_img_dir = img_val_dir if in_val[i] else img_dir  # rester sur place pour train
        dest_lbl_dir = lbl_val_dir if in_val[i] else lbl_dir

        if in_val[i]:
            # déplacer vers val
            shutil.move(ip, os.path.join(dest_img_dir, os.path.basename(ip)))
            if os.path.exists(lp):
                shutil.move(lp, os.path.join(dest_lbl_dir, os.path.basename(lp)))
            else:
                # créer un .txt vide si manquant, pour cohérence
                open(os.path.join(dest_lbl_dir, name + ".txt"), "a").close()
            moved_val += 1
        else:
            # s'assurer que le label existe (éventuellement vide)
            if not os.path.exists(lp):
                open(lp, "a").close()
            moved_train += 1

    # 8) Petit bilan
    def count_dir(lbl_path):
        cnt = Counter()
        for p in glob(os.path.join(lbl_path, "*.txt")):
            with open(p) as f:
                for line in f:
                    line=line.strip()
                    if line:
                        try: cnt[int(line.split()[0])] += 1
                        except: pass
        return cnt

    val_counts   = count_dir(lbl_val_dir)
    train_counts = count_dir(lbl_dir)

    HF_log.info(f"Split done. val images={moved_val}, train images={moved_train}")
    HF_log.info(f"Per-class (train): {dict(train_counts)}")
    HF_log.info(f"Per-class (val)  : {dict(val_counts)}")
