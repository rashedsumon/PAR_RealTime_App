import streamlit as st
import torch
from PIL import Image
from data_loader import download_dataset, get_transforms
from model import MultiAttributeModel
from utils import predict_attributes

st.title("Passenger Attribute Recognition (PAR) System")

# Load dataset (optional, for reference)
data_path = download_dataset()

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiAttributeModel()
model.to(device)

# Upload image
uploaded_file = st.file_uploader("Upload cropped passenger image", type=["jpg","png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess
    transform = get_transforms()
    img_tensor = transform(image)

    # Predict attributes
    predictions = predict_attributes(model, img_tensor, device)
    st.subheader("Predicted Attributes:")
    for attr, value in predictions.items():
        st.write(f"{attr}: {value}")
