# HFinder

HFinder is a modular and semi-automated pipeline for the detection and segmentation of filamentous microbes (such as fungi or oomycetes) and their specialized infection structures (e.g., haustoria) in confocal microscopy images. Designed for flexibility and extensibility, it supports both automated detection using YOLOv8 and manual correction workflows when needed.

The pipeline handles:

- Image preprocessing and composite generation from multi-channel TIFF files.
- Binary mask creation and contour extraction for filamentous structures.
- Conversion to YOLO-compatible annotation formats for training object detection models.
- Training and evaluation of segmentation models via Ultralytics YOLOv8.
- Integration of external annotations (e.g., from Makesense AI) with rescaling support.

HFinder is intended for researchers studying plant–microbe interactions and host colonization dynamics, providing tools to accelerate the annotation and analysis of large image datasets.
