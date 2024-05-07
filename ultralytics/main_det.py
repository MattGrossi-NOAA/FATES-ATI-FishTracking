from ultralytics import YOLO

# Create a new YOLO model from scratch
#model = YOLO('yolov8l.yaml')
model = YOLO('yolov8n.yaml')
#model = YOLO('yolov8m.yaml')
#model = YOLO('yolov8s.yaml')
#model = YOLO('yolov8x.yaml')
# Load a pretrained YOLO model (recommended for training)
#model = YOLO('yolov8n.pt')

# Train the model
results = model.train(data='fish.yaml', epochs=60, device=[6,7])
#multi GPU
#$ python -m torch.distributed.launch --nproc_per_node 2 main_det.py --batch-size 64 --data coco.yaml --weights yolov5s.pt
