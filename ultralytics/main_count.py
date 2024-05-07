from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2


model = YOLO("best.pt")
cap =cv2.VideoCapture("fish.avi")
#model = YOLO("best.pt")
#cap =cv2.VideoCapture("fish.avi")
assert cap.isOpened(), "Error reading video file"

# Get the dimensions of the video frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the region points based on the video dimensions
# Here, we're using the entire video frame, but this can be adjusted
expanded_region_points = [
    (0, 0),                              # Top-left corner
    (frame_width, 0),                     # Top-right corner
    (frame_width, frame_height),          # Bottom-right corner
    (0, frame_height)                     # Bottom-left corner
]
# Video Writer
video_writer = cv2.VideoWriter("ultralytics_object_counting.avi",
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                int(cap.get(5)),
                                (int(cap.get(3)), int(cap.get(4))))

counter = object_counter.ObjectCounter()  # Init Object Counter
#region_points = [(20, 400), (1500, 20), (1500, 1200), (820, 1080)]


counter.set_args(view_img=True,
                 reg_pts=expanded_region_points,
                 classes_names=model.names,
                 draw_tracks=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    tracks = model.track(im0, persist=True, show=False)
    im0 = counter.start_counting(im0, tracks)

    # Access and display in/out counts
    in_count = getattr(counter, 'in_counts', 0)
    out_count = getattr(counter, 'out_counts', 0)

    cv2.putText(im0, f"Total In : {in_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(im0, f"Total Out: {out_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv2.LINE_AA)

    video_writer.write(im0)

video_writer.release()
"""while cap.isOpened():
    success, frame = cap.read()
    if not success:
        exit(0)
    tracks = model.track(frame, persist=True, show=False)
    frame = counter.start_counting(frame, tracks)
    #video_writer.write(frame)

#video_writer.release()"""
