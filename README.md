# HFinder

HFinder is a modular and extensible pipeline for the detection, segmentation, 
and analysis of filamentous microbes (e.g., fungi, oomycetes) and their 
infection structures (such as haustoria) in confocal microscopy images. Built 
around [YOLOv8](https://yolov8.com/), it provides a complete workflow from raw 
multi-channel TIFFs to consolidated predictions and ready-to-use annotations.

The pipeline integrates both automated machine learning and manual correction, 
making it suitable for large-scale studies as well as expert-guided analysis. 
It supports dataset generation, model training, multi-fusion prediction, and 
result consolidation into formats compatible with external tools such as 
[Makesense AI](https://www.makesense.ai/).

## Key Features
- **Multi-channel TIFF processing**: Automatically generates fused RGB 
composites from arbitrary channel combinations, reproducing training 
conditions and enabling robust ensembling.
- **Flexible annotation workflows**: Combine automated segmentation, YOLO-based 
detection, and manual polygon annotations. External annotations can be imported 
and rescaled seamlessly.
- **End-to-end YOLOv8 support**: Create YOLO-ready datasets, train new models, 
evaluate performance, and export predictions in both consolidated JSON and 
COCO JSON formats (containing boxes and polygons).
- **Prediction with ensembling and voting**: Runs detection across multiple 
channel fusions, then consolidates results with intra- and inter-class IoU 
voting strategies, ensuring consistent and biologically meaningful predictions.
- **Class-aware customization**: Define per-class segmentation strategies, 
allow or forbid overlaps between classes, and control prediction thresholds for 
fine-grained analysis.
- **Integration with external tools**: Results can be directly visualized and 
corrected in Makesense.ai or other COCO-compatible annotation editors.

HFinder is designed for researchers studying plant–microbe interactions and 
host colonization dynamics, providing a reproducible and semi-automated 
framework to accelerate annotation, training, and large-scale image analysis.
It provides both **pretrained models** and the **full workflow** to retrain on 
custom datasets, making it immediately usable yet fully adaptable.

## Installation

### Dependencies
To run HFinder, you need to install the following dependencies:
- The YAML library [pyyaml](https://pypi.org/project/PyYAML/)
- The Python YOLO interface [ultralytics](https://docs.ultralytics.com/fr/quickstart/)
- The TIFF library [tifffile](https://pypi.org/project/tifffile/)
- The Python image processing library [scikit-image](https://scikit-image.org/)

You can do so by running:
```
pip install pyyaml ultralytics tifffile scikit-image
```
It is recommended to use a [virtual environment](https://docs.python.org/3/library/venv.html).

### Running
You can then run HFinder as follows:

```
$ python hfinder.py <action>
```

where action can be `preprocess`, `train`, or `predict`.

|Action|Description|
|-|-|
|`preprocess`|Runs image segmentation only. Useful for troubleshooting training dataset generation.|
|`train`|Performs image segmentation, then trains a YOLOv8 model.|
|`predict`|Detects instances and generates polygons for the specified classes.|


## Training Mode

See [Training guide](README.train.md).

## Prediction Mode

See [Prediction guide](README.predict.md).

## Technical Introduction to HFinder

For a more advanced, programmer-oriented overview of HFinder, see [Technical guide](README.technical.md).

