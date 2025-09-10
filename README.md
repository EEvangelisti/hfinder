# HFinder

HFinder is a modular and extensible pipeline for the detection, segmentation, 
and analysis of filamentous microbes (e.g., fungi, oomycetes) and their 
infection structures (such as haustoria) in confocal microscopy images. Built 
around [YOLOv8](https://yolov8.com/), it provides a complete workflow from raw 
multi-channel TIFFs to consolidated predictions and ready-to-use annotations.
The pipeline supports dataset generation, model training, multi-fusion prediction, and 
result consolidation into formats compatible with external tools such as 
[Makesense AI](https://www.makesense.ai/).


## Installation

HFinder is distributed as a Python package and can be installed with `pip`.

See the [Installation guide](INSTALL.md) for detailed instructions
(including GPU/CPU support).



## Running
You can then run HFinder as follows:

```
$ hfinder <action>
```

where action can be `preprocess`, `train`, or `predict`.

|Action|Description|
|-|-|
|`preprocess`|Runs image segmentation only. Useful for troubleshooting training dataset generation.|
|`train`|Performs image segmentation, then trains a YOLOv8 model.|
|`predict`|Detects instances and generates polygons for the specified classes.|


### Training mode

Use training mode if you want to train HFinder to recognize additional structures.  
For step-by-step instructions, see [Training guide](doc/training.md).

You can download the official training dataset from Zenodo:  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.17091805)


### Prediction mode

Use prediction mode to run inference with HFinder.

A pre-trained model is required:
- You can train your own model (see [Training mode](#training-mode) above).  
- Or you can download the official pre-trained weights from Zenodo:  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.17090880)

For step-by-step instructions, see [Prediction guide](doc/prediction.md).


### HFinder Toolbox

HFinder includes additional tools to explore predictions, analyze signals, and measure distances.  
For an introduction to these auxiliary scripts, see the [Toolbox guide](doc/toolbox.md).
