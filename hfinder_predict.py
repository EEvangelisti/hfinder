# This is a preliminary version of the prediction pipeline

from ultralytics import YOLO

# Load a pretrained YOLOv8 segmentation model
model = YOLO("best.pt")  # n = nano, small & fast; can use yolov8s-seg.pt, yolov8m-seg.pt, etc.

# Run prediction on an image, directory, or video
results = model.predict(
    source="GFP-E5 N2.merge.jpg",  # path to file, directory, URL, or device index (e.g., 0 for webcam)
    save=True,           # save results to 'runs/predict-seg'
    conf=0.25             # confidence threshold
)

# Iterate over results if you need masks, boxes, or class info
for r in results:
    masks = r.masks.data if r.masks is not None else None
    boxes = r.boxes.xyxy if r.boxes is not None else None
    print(f"Detected {len(r.boxes)} objects")
