# Predicting Microscopic Structures with HFinder

## How does it work?

(...)

## Parameters


|Command|Description|Default value|
|-|-|-|
|`-d <path>` or `--tiff_dir <path>`|Path to the folder containing the training image files or the images to predict|data|
|`-de <str>` or `--device <str>`|Computation device for PyTorch (e.g. 'cpu', '0' for GPU)|cpu|
|`-vi <float>` or `--vote_iou <float>`|IoU threshold for merging overlapping detections during voting|0.5|
|`-vm <int>` or `--vote_min <int>`|Minimum number of votes required to keep a consolidated detection|2|
|`-b <int>` or `--batch <int>`|Batch size used for running YOLO predictions|8|
|`-y <path>` or `--yaml <path>`|Path to the training dataset.yaml (used to resolve class names)|dataset.yaml|
|`-w <path>` or `--weights <path>`|Weights for prediction|yolov8n-seg.pt|
|`-dbg` or `--debug`|Enable debug mode to display additional logs and diagnostic information|Not activated|


