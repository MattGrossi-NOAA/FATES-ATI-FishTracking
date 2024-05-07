from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("best.pt")  # load a pretrained model (recommended for training)

# Use the model
metrics = model.val()  # evaluate model performance on the validation set

