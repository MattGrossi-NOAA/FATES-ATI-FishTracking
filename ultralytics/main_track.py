from ultralytics import YOLO

# Load an official or custom model
model = YOLO('best.pt')  # Load an official Detect model

# Perform tracking with the model
results = model.track(source="fish.avi", save=True, conf=0.5, iou=0.5, show=True, tracker="bytetrack.yaml")  # Tracking with default tracker
#results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker

