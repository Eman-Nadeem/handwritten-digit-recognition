import streamlit as st
import torch
import torch.nn as nn
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Handwritten Digit Prediction",
    layout="wide"
)

st.title("Handwritten Digit Prediction")
st.write(
    "Upload a handwritten digit image and let "
    "CNN and Vision Transformer predict it."
)


# ==========================================
# CNN MODEL
# ==========================================

class DigitCNN(nn.Module):

    def __init__(self):
        super(DigitCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# ==========================================
# VIT MODEL
# ==========================================

class PatchEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 16)

    def forward(self, patches):
        return self.projection(patches)


class PositionalEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.position = nn.Parameter(
            torch.zeros(1, 16, 16)
        )

    def forward(self, x):
        return x + self.position


class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, x):
        return self.transformer(x)


class DigitClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = x.mean(dim=1)
        return self.fc(x)


class DigitViT(nn.Module):

    def __init__(self):
        super().__init__()

        self.patch_embedding = PatchEmbedding()
        self.position_embedding = PositionalEmbedding()
        self.transformer = TransformerBlock()
        self.classifier = DigitClassifier()

    def forward(self, x):

        x = x.unfold(1, 2, 2).unfold(2, 2, 2)
        x = x.reshape(x.shape[0], 16, 4)

        x = self.patch_embedding(x)
        x = self.position_embedding(x)
        x = self.transformer(x)
        x = self.classifier(x)

        return x


# ==========================================
# LOAD TRAINED MODELS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_models():

    cnn = DigitCNN()

    cnn.load_state_dict(
        torch.load(
            BASE_DIR / "models" / "cnn_model.pth",
            map_location="cpu"
        )
    )

    cnn.eval()

    vit = DigitViT()

    vit.load_state_dict(
        torch.load(
            BASE_DIR / "models" / "vit_model.pth",
            map_location="cpu"
        )
    )

    vit.eval()

    return cnn, vit


try:
    cnn_model, vit_model = load_models()

except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    # Put transparent PNGs on a white background first
    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image.convert("RGBA"))

    # Convert to grayscale
    image = image.convert("L")

    # Invert only if the background is bright
    # (training digits are bright strokes on a dark background)
    if np.array(image).mean() > 127:
        image = ImageOps.invert(image)

    # Ignore faint JPEG noise when finding the digit and rejecting blanks
    arr = np.array(image)
    mask = Image.fromarray(((arr > 40) * 255).astype(np.uint8))
    bbox = mask.getbbox()

    if bbox is None:
        raise ValueError("Blank image! Please upload a digit.")

    # Crop around the digit
    image = image.crop(bbox)

    # Center digit on a square canvas with a small margin,
    # like the sklearn digits
    w, h = image.size
    side = int(max(w, h) * 1.2)

    canvas = Image.new("L", (side, side), 0)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))

    # Resize to 8x8
    resized = canvas.resize(
        (8, 8),
        Image.Resampling.LANCZOS
    )

    # Normalize to 0-1 and stretch contrast so the stroke reaches 1.0
    arr = np.array(resized, dtype=np.float32)
    arr = arr / max(arr.max(), 1e-6)

    return resized, arr


# ==========================================
# IMAGE UPLOAD
# ==========================================

st.subheader("Upload Your Handwritten Digit")

uploaded_file = st.file_uploader(
    "Choose a PNG or JPEG image",
    type=["png", "jpg", "jpeg"]
)


# ==========================================
# PREDICTIONS
# ==========================================

if uploaded_file is not None:

    try:

        original_image = Image.open(uploaded_file)

        processed_image, image_array = preprocess_image(
            original_image
        )

        st.subheader("Image Preview")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Original Image")
            st.image(
                original_image,
                width=160
            )

        with col2:
            st.write("Processed 8×8 Image")
            st.image(
                processed_image.resize(
                    (240, 240),
                    Image.Resampling.NEAREST
                ),
                width=160
            )

        # Prepare CNN input: [1, 1, 8, 8]
        image_tensor = torch.tensor(
            image_array,
            dtype=torch.float32
        )

        cnn_input = image_tensor.unsqueeze(0).unsqueeze(0)

        # Prepare ViT input: [1, 8, 8]
        vit_input = image_tensor.unsqueeze(0)

        # Make predictions
        with torch.no_grad():

            cnn_output = cnn_model(cnn_input)
            vit_output = vit_model(vit_input)

            cnn_prediction = torch.argmax(
                cnn_output,
                dim=1
            ).item()

            vit_prediction = torch.argmax(
                vit_output,
                dim=1
            ).item()

        # Display predictions
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "CNN Prediction",
                cnn_prediction
            )

        with col2:
            st.metric(
                "ViT Prediction",
                vit_prediction
            )

        # Compare models
        st.subheader("Model Comparison")

        if cnn_prediction == vit_prediction:

            st.success(
                f"Both models agree! Predicted digit: "
                f"{cnn_prediction}"
            )

        else:

            st.warning(
                "The models disagree. "
                "They predicted different digits."
            )

    except Exception as e:

        st.error(f"Image processing error: {e}")

else:

    st.info("Please upload a PNG or JPEG image of one handwritten digit.")