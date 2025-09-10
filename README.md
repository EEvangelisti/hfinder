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

HFinder is distributed as a Python package and can be installed with `pip`.

See the [Installation guide](INSTALL.md) for detailed instructions
(including GPU/CPU support).


## Pre-trained weights

You can download the pre-trained weights from Zenodo below

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.17090880)


## Running
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


### Training Mode

See [Training guide](doc/training.md).

### Prediction Mode

See [Prediction guide](doc/prediction.md).

### HFinder Toolbox

For an introduction to HFinder auxiliary scripts, see [Toolbox guide](doc/toolbox.md).

