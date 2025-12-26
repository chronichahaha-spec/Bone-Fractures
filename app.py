import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

#Page Setting
st.set_page_config(page_title="Bone Fractures Detection", layout="wide")

st.title("Bone Fractures CT Detection System")
st.write("Upload your CT JPG image and get automatic lesion detection.")

@st.cache_resource
def load_model():
    return YOLO("YOLO.pt")

model = load_model()

#User Upload Image
uploaded_file = st.file_uploader("Upload CT JPG Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded CT Image", use_column_width=True)

    if st.button("Run YOLO Detection"):

        with st.spinner("Detecting..."):

            #Save as Temporary Image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name

            #YOLO Detection
            results = model(tmp_path, conf=0.25)

            #Plot Detection Result
            res_plotted = results[0].plot()

            # BGR->RGB
            res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            st.success("Detection Finished!")
            st.image(res_plotted, caption="Detection Result", use_column_width=True)

            os.remove(tmp_path)
