# HFinder

HFinder is a modular and semi-automated pipeline for the detection and segmentation of filamentous microbes (such as fungi or oomycetes) and their specialized infection structures (e.g., haustoria) in confocal microscopy images. Designed for flexibility and extensibility, it supports both automated detection using [YOLOv8](https://yolov8.com/) and manual correction workflows when needed.

The pipeline handles:

- Image preprocessing and composite generation from multi-channel TIFF files.
- Binary mask creation and contour extraction for filamentous structures.
- Conversion to YOLO-compatible annotation formats for training object detection models.
- Training and evaluation of segmentation models via Ultralytics YOLOv8.
- Integration of external annotations (e.g., from [Makesense AI](https://www.makesense.ai/)) with rescaling support.

HFinder is intended for researchers studying plant–microbe interactions and host colonization dynamics, providing tools to accelerate the annotation and analysis of large image datasets.

## Key Features
- **Multi-channel TIFF support**: Automatically extracts, thresholds, or segments specific channels within multi-frame TIFFs. Each channel or z-slice can be treated independently or combined based on user-defined instructions.
- **Flexible annotation workflow**: Supports both automatic segmentation (via thresholding) and custom annotations (via JSON polygons), enabling hybrid workflows.
- **YOLOv8-compatible dataset generation**: Converts raw TIFF data into structured datasets (images, masks, metadata) for direct use in YOLOv8 training pipelines, including automatic generation of dataset.yaml.
- **Class-aware image mapping**: Supports class-specific instructions, allowing different segmentation strategies per class or per image.


## Installation

### Dependencies
To run HFinder, you need to install the following dependencies:
- The YAML library [pyyaml](https://pypi.org/project/PyYAML/)
- The Python YOLO interface [ultralytics](https://docs.ultralytics.com/fr/quickstart/)
- The TIFF library [tifffile](https://pypi.org/project/tifffile/) 

It is recommended to use a virtual environment.
```
pip install pyyaml ultralytics tifffile
```

### Running
You can then run HFinder as follows:

```
$ python hfinder.py <action>
```

where action can be `check`, `train`, or `predict`.

## Training mode

(...)

## Prediction mode

(...)
