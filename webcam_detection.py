from ultralytics import YOLO
import cv2


# Load YOLO model
model = YOLO("yolov8n.pt")


# Start webcam
camera = cv2.VideoCapture(0)


while True:

    success, frame = camera.read()

    if not success:
        break


    # Detect and track objects

    results = model.track(
        frame,
        persist=True
    )


    # Draw bounding boxes

    detected_frame = results[0].plot()


    # Display output

    cv2.imshow(
        "YOLO Object Detection",
        detected_frame
    )


    # Press q to exit

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()

cv2.destroyAllWindows()