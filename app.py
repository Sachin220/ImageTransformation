import streamlit as st
import cv2
import numpy as np
from PIL import Image
import transformations  # Ensure this file exists in the same directory

# Title of the app
st.title("Affine Transformation App")

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

# If a file is uploaded
if uploaded_file:
    try:
        # Load the image
        image = np.array(Image.open(uploaded_file))

        # Display the uploaded image
        st.image(image, caption="Original Image", use_column_width=True)

        # Transformation options
        transformation = st.selectbox(
            "Choose a transformation",
            ["Translate", "Scale", "Rotate", "Shear"]
        )

        # Apply transformations based on user selection
        if transformation == "Translate":
            tx = st.slider("Translate X (pixels)", -200, 200, 50)
            ty = st.slider("Translate Y (pixels)", -200, 200, 50)
            transformed_image = transformations.translate(image, tx, ty)

        elif transformation == "Scale":
            sx = st.slider("Scale X (factor)", 0.1, 3.0, 1.0)
            sy = st.slider("Scale Y (factor)", 0.1, 3.0, 1.0)
            transformed_image = transformations.scale(image, sx, sy)

        elif transformation == "Rotate":
            angle = st.slider("Rotation Angle (degrees)", 0, 360, 45)
            transformed_image = transformations.rotate(image, angle)

        elif transformation == "Shear":
            shear_factor = st.slider("Shear Factor", -1.0, 1.0, 0.0)
            transformed_image = transformations.shear(image, shear_factor)

        # Display the transformed image
        st.image(transformed_image, caption=f"Transformed Image ({transformation})", use_column_width=True)

    except Exception as e:
        st.error(f"Error occurred: {e}")
else:
    st.info("Please upload an image to get started.")
