# Predicting Microscopic Structures with HFinder

## How does it work?

(...)

## Parameters


|Short|Long|Description|Default value|
|-|-|-|-|
|`-d <path>`|`--tiff_dir <path>`|Path to the folder containing the training image files or the images to predict|data|
|`-de <str>`|`--device <str>`|Computation device for PyTorch (e.g. 'cpu', '0' for GPU)|cpu|

```
usage: hfinder predict [-h] [-d TIFF_DIR] [-de DEVICE] [-vi VOTE_IOU]
                       [-vm VOTE_MIN] [-b BATCH] [-y YAML] [-w WEIGHTS] [-dbg]

options:
  -d <path>, --tiff_dir <path>
                        Path to the folder containing the training image files
                        or the images to predict (default: data)
  -de <str>, --device <str>
                        Computation device for PyTorch (e.g. 'cpu', '0' for
                        GPU). (default: cpu)
  -vi <float>, --vote_iou <float>
                        IoU threshold for merging overlapping detections
                        during voting. (default: 0.5)
  -vm <int>, --vote_min <int>
                        Minimum number of votes required to keep a
                        consolidated detection. (default: 2)
  -b <int>, --batch <int>
                        Batch size used for running YOLO predictions.
                        (default: 8)
  -y <path>, --yaml <path>
                        Path to the training dataset.yaml (used to resolve
                        class names). (default: dataset.yaml)
  -w <path>, --weights <path>
                        Weights for prediction. (default: yolov8n-seg.pt)
  -dbg, --debug         Enable or disable debug mode to display additional
                        logs and diagnostic information.
```


