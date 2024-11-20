import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image

# Helper functions
def load_image(file):
    """Load an image and convert it to a tensor."""
    image = Image.open(file).convert("RGB")
    return to_tensor(image).unsqueeze(0)  # Convert to [1, C, H, W] tensor

def show_image(tensor):
    """Convert a tensor to a PIL image."""
    image = tensor.squeeze(0)  # Remove batch dimension
    return to_pil_image(image)

def get_affine_matrix(translation=(0, 0), rotation=0, scale=(1, 1), shear=(0, 0)):
    """Generate a 2x3 affine transformation matrix."""
    angle = np.radians(rotation)
    shear_x, shear_y = np.radians(shear)

    # Rotation matrix
    rot_matrix = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ], dtype=torch.float32)

    # Translation matrix
    trans_matrix = torch.tensor([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ], dtype=torch.float32)

    # Scaling matrix
    scale_matrix = torch.tensor([
        [scale[0], 0, 0],
        [0, scale[1], 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    # Shearing matrix
    shear_matrix = torch.tensor([
        [1, np.tan(shear_x), 0],
        [np.tan(shear_y), 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    # Combine transformations
    affine_matrix = trans_matrix @ rot_matrix @ scale_matrix @ shear_matrix
    return affine_matrix[:2, :]  # Return 2x3 matrix for affine_grid

def apply_transformation(image, matrix):
    """Apply an affine transformation to an image."""
    _, C, H, W = image.shape
    grid = F.affine_grid(matrix.unsqueeze(0), size=image.size(), align_corners=False)
    transformed_image = F.grid_sample(image, grid, align_corners=False)
    return transformed_image

# Streamlit App
st.title("Image Transformation App")
st.write("Upload an image and apply transformations using predefined settings.")

# Load transformations
try:
    transformations = torch.load("model.pth")
except FileNotFoundError:
    st.error("Transformation model file ('model.pth') not found. Please provide it in the working directory.")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    try:
        image = load_image(uploaded_file)
        st.image(show_image(image), caption="Original Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # Apply transformations
    st.subheader("Transformed Images")
    for transform in transformations:
        try:
            matrix = get_affine_matrix(**transform["params"])
            transformed_image = apply_transformation(image, matrix)

            st.image(
                show_image(transformed_image),
                caption=f"{transform['name']} Applied",
                use_column_width=True
            )
        except Exception as e:
            st.error(f"Error applying transformation {transform['name']}: {e}")

st.write("Customize transformations or add more in the `model.pth` file.")
