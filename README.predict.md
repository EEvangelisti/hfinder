# Predicting Microscopic Structures with HFinder

## How does it work?

(...)

## Parameters


|Command|Description|Default value|
|-|-|-|
|`-d <path>`<br>`--tiff_dir <path>`|Path to the folder containing the training image files or the images to predict|data|
|`-de <str>`<br>`--device <str>`|Computation device for PyTorch (e.g. 'cpu', '0' for GPU)|cpu|
|`-vi <float>`<br>`--vote_iou <float>`|IoU threshold for merging overlapping detections during voting|0.5|
|`-vm <int>`<br>`--vote_min <int>`|Minimum number of votes required to keep a consolidated detection|2|
|`-b <int>`<br>`--batch <int>`|Batch size used for running YOLO predictions|8|
|`-y <path>`<br>`--yaml <path>`|Path to the training dataset.yaml (used to resolve class names)|dataset.yaml|
|`-w <path>`<br>`--weights <path>`|Weights for prediction|yolov8n-seg.pt|
|`-dbg`<br>`--debug`|Enable debug mode to display additional logs and diagnostic information|Not activated|


