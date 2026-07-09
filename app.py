# from ultralytics import YOLO
# import cv2


# # loading pretrained YOLO model

# model = YOLO("yolov8n.pt")


# # image path

# image_path = "input/sample.jpg"


# # read image

# image = cv2.imread(image_path)


# # detect objects

# results = model(image)


# # draw bounding boxes

# output = results[0].plot()


# # save output

# cv2.imwrite(
#     "output/detected_image.jpg",
#     output
# )


# print("Object detection completed")

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np


# --------------------------------
# Page Setup
# --------------------------------

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 Real-Time Object Detection System")

st.write(
    "AI & Data Science Internship Project - Horizon TechX"
)

st.divider()


# --------------------------------
# Load YOLO Model
# --------------------------------

@st.cache_resource
def load_model():

    model = YOLO("yolov8n.pt")

    return model


model = load_model()


# --------------------------------
# Sidebar
# --------------------------------

st.sidebar.title("Settings")

confidence = st.sidebar.slider(
    "Confidence Level",
    0.1,
    1.0,
    0.5
)


option = st.sidebar.radio(
    "Choose Detection Mode",
    [
        "Image Detection",
        "Webcam Detection"
    ]
)


# --------------------------------
# IMAGE DETECTION
# --------------------------------

if option == "Image Detection":

    st.subheader("Upload Image")

    uploaded_image = st.file_uploader(
        "Choose Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_image:

        image = Image.open(uploaded_image)

        st.image(
            image,
            caption="Original Image"
        )


        image_array = np.array(image)


        if st.button("Detect Objects"):


            result = model(
                image_array,
                conf=confidence
            )


            output_image = result[0].plot()


            st.subheader(
                "Detection Result"
            )

            st.image(
                output_image
            )


            detected = result[0].boxes


            st.success(
                f"Objects detected: {len(detected)}"
            )


# --------------------------------
# WEBCAM DETECTION INFO
# --------------------------------

else:

    st.subheader("📹 Live Webcam Detection")

    start = st.button("Start Camera")

    frame_window = st.image([])

    if start:

        camera = cv2.VideoCapture(0)

        while True:

            success, frame = camera.read()

            if not success:
                st.error("Camera not detected")
                break


            results = model.track(
                frame,
                persist=True,
                conf=confidence
            )


            output_frame = results[0].plot()


            output_frame = cv2.cvtColor(
                output_frame,
                cv2.COLOR_BGR2RGB
            )


            frame_window.image(
                output_frame
            )


        camera.release()

# --------------------------------
# Footer
# --------------------------------

st.divider()

st.write(
    "Developed using Python, OpenCV and YOLOv8"
)