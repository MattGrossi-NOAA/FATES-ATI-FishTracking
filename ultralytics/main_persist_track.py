import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8l.pt")

# Open the video file
video_path = "fish.avi"
cap = cv2.VideoCapture(video_path)

# Define the output video file path
output_video_path = "output_video.avi"

# Get the video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video file
        out.write(annotated_frame)

        # Break the loop if the end of the video is reached
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object, VideoWriter object, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

