# HFinder

HFinder is a modular and extensible framework for the detection, segmentation, 
and quantitative analysis of filamentous microbes (e.g. fungi, oomycetes) and 
their infection structures (such as haustoria) in confocal microscopy images.

The framework is designed to address the specific challenges posed by 
plant-microbe interfaces, including sparse, anisotropic, and morphologically 
heterogeneous structures imaged across multiple fluorescence channels. 
Predictions are performed independently on individual channels and subsequently 
reconciled through a biologically informed decision scheme, enabling 
channel-aware quantitative analyses.

Built on a [YOLOv8](https://yolov8.com/)-based object detection backbone, HFinder provides an 
end-to-end workflow from raw multi-channel TIFF images to consolidated 
predictions and ready-to-use annotations. The framework supports dataset 
generation, model training, multi-model fusion, and result consolidation 
into formats compatible with external annotation and visualization tools such 
as [Makesense AI](https://www.makesense.ai/).


## Installation

HFinder is distributed as a Python package and can be installed with `pip`.

See the [Installation guide](INSTALL.md) for detailed instructions
(including GPU/CPU support).



## Command-line usage

After installation, HFinder provides a command-line interface.  
Run it with the following syntax:

```
hfinder <action>
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
For step-by-step instructions, see [Prediction guide](doc/prediction.md).

A pre-trained model is required:
- You can train your own model (see [Training mode](#training-mode) above).  
- Or you can download the official pre-trained weights from Zenodo:  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.17090880)




### HFinder Toolbox

HFinder includes additional tools to explore predictions, analyze signals, and measure distances.  
For an introduction to these auxiliary scripts, see the [Toolbox guide](doc/toolbox.md).
