import streamlit as st
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet34
import torch.nn.functional as F
import tifffile

import numpy as np

# Define the transformation to apply to input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to make predictions
def predict(image):
    # Convert NumPy array to PIL Image if the input is a NumPy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = F.softmax(model(image), dim=1)
    return output

if __name__ == '__main__':
    # Load the pre-trained ResNet34 model
    model_path = "resnet34--epoch=16-val_acc=0.71-val_loss=0.63.ckpt"
    model = resnet34(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)  # Assuming model has 2 output classes (0 and 1)

    # Print keys of the loaded state_dict
    loaded_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    print("Loaded State Dict Keys:", loaded_state_dict.keys())

    # Filter out unnecessary keys
    filtered_state_dict = {k: v for k, v in loaded_state_dict['state_dict'].items() if k in model.state_dict().keys()}

    # Load the filtered state_dict
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()

    # Streamlit app
    st.title("Predictions Leukemia")
    uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'tif', 'tiff'])

    if uploaded_file is not None:
        # Check if the uploaded file is a TIFF file
        if uploaded_file.name.lower().endswith(('.tif', '.tiff')):
            # Use tifffile to open TIFF images
            image = tifffile.imread(uploaded_file)
        else:
            # For other image types, use PIL
            image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Make predictions
        predictions = predict(image)

        # Display the results
        st.subheader("Predictions:")
        st.write(f"Probability of being healthy (Y=0): {predictions[0][0]:.4f}")
        st.write(f"Probability of having blast cells (Y=1): {predictions[0][1]:.4f}")