import streamlit as st
import torch
import tifffile
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image

# Define the transformation to apply to input images
def preprocess_image(image):
    # If the input is a NumPy array, convert it to a PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # Define a sequence of transformations (resize and convert to tensor)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    # Apply transformations and add a batch dimension
    return transform(image).unsqueeze(0)

# Function to make predictions
def predict(model, image):
    # Disable gradient calculation during inference
    with torch.no_grad():
        # Apply softmax to model predictions
        output = F.softmax(model(image), dim=1)
    return output

# Function to load and prepare the model
def load_model(model_path):
    # Create a ResNet34 model with 2 output classes (0 and 1)
    model = resnet34(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)

    # Load the pre-trained weights
    loaded_state_dict = torch.load(
        model_path, map_location=torch.device("cpu"))

    # Filter unnecessary keys from the loaded state_dict
    filtered_state_dict = {
        k: v
        for k, v in loaded_state_dict["state_dict"].items()
        if k in model.state_dict().keys()
    }
    # Load the filtered state_dict into the model
    model.load_state_dict(filtered_state_dict, strict=False)
    # Set the model to evaluation mode
    model.eval()

    return model

if __name__ == "__main__":
    # Load the pre-trained ResNet34 model
    model_path = "resnet34--epoch=16-val_acc=0.71-val_loss=0.63.ckpt"
    model = load_model(model_path)

    # Streamlit app
    st.title("Leukemia Predictions")
    st.write(
        "Welcome to the Leukemia Predictions Demo. Upload an image to get predictions."
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file", type=["png", "jpg", "tif", "tiff"]
    )

    # Handle uploaded file
    if uploaded_file is not None:
        try:
            # Check if the uploaded file is a TIFF file
            if uploaded_file.name.lower().endswith((".tif", ".tiff")):
                # Use tifffile to open TIFF images
                image = tifffile.imread(uploaded_file)
            else:
                # For other image types
                image = Image.open(uploaded_file)

            # Display the uploaded image
            st.image(
                image,
                caption="Uploaded Image.",
                use_column_width=True,
                output_format="JPEG",
                channels="RGB",
                clamp=True,
            )

            # Make predictions
            image_tensor = preprocess_image(image)
            predictions = predict(model, image_tensor)

            # Set a threshold for predictions
            threshold = 0.8

            # Display the results
            st.subheader("Predictions:")

            # Format prediction probabilities
            prob_healthy = predictions[0][0].item()
            prob_blast_cells = predictions[0][1].item()

            # Display predictions
            st.write(f"Probability of being healthy: {prob_healthy:.2%}")
            st.write(f"Probability of having blast cells: {prob_blast_cells:.2%}")

            # Identify patients with highly likely leukemia
            if prob_blast_cells >= threshold:
                st.error("Patient highly likely to have leukemia. Recommend urgent referral to a healthcare professional.")
            # low risk
            else:
                st.success("Patient likely not at risk of leukemia based on model predictions. Continue routine monitoring.")

        except Exception:
            st.warning("This may not be a picture of blood cell samples. Please upload a new picture.")