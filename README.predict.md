# Object Detection and Segmentation with HFinder

## Running Predictions

HFinder includes a prediction module that allows you to run object detection and
segmentation on new microscopy images. The process is designed to be simple and
consistent with the training workflow:

1. **Provide input TIFF files**  
   Place your microscopy images (single-frame or multi-frame TIFFs) in the 
   directory specified by `-d ` or `--tiff_dir` (see parameter list below).

2. **Load the trained model and dataset YAML**  
   Predictions require a trained model and the description of the classes to 
   detect. 
   - Specify the path to the trained YOLO weights (`.pt` file) obtained during
     training using `-w` or `--weights`.
   - Provide the same dataset `.yaml` file used during training using `-y` or `--yaml`.
     This file defines the list of classes and ensures predictions can be mapped
     back to meaningful labels.

3. **Channel fusion and ensembling**  
   For each TIFF, HFinder generates multiple RGB images by combining channels in
   different ways. Predictions are run on all these images, and results are
   consolidated to reduce noise and highlight consistent detections.

4. **Export results**  
   Predictions are saved in a dedicated output folder under `runs/predict/`.
   - Each TIFF produces a set of JPEG images with overlays showing the detections.
   - A `consolidated.json` file summarizes the detections per TIFF.
   - A `coco.json` file is also generated, containing the results in COCO format
   (bounding boxes and polygons), ready to be viewed or edited with tools such 
   as [Makesense.ai](https://www.makesense.ai/).

---

This workflow allows you to quickly apply a trained model to new data, explore 
results visually, and refine annotations if needed.

## Parameter list

|Command|Description|Default value|
|-|-|-|
|`-d <path>` or<br>`--tiff_dir <path>`|Path to the folder containing the training image files or the images to predict|data|
|`-w <path>` or<br>`--weights <path>`|Weights for prediction|yolov8n-seg.pt|
|`-y <path>` or<br>`--yaml <path>`|Path to the training dataset.yaml (used to resolve class names)|dataset.yaml|
|`-b <int>` or<br>`--batch <int>`|Batch size used for running YOLO predictions|8|
|`-dbg` or<br>`--debug`|Enable debug mode to display additional logs and diagnostic information|Inactive|
|`-de <str>` or<br>`--device <str>`|Computation device for PyTorch (e.g. 'cpu', '0' for GPU)|cpu|
|`-vi <float>` or<br>`--vote_iou <float>`|IoU threshold for merging overlapping detections during voting|0.5|
|`-vm <int>` or<br>`--vote_min <int>`|Minimum number of votes required to keep a consolidated detection|2|

