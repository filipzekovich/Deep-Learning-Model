import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Set page config
st.set_page_config(
    page_title="Fashion-MNIST Classifier",
    page_icon="👕",
    layout="wide"
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


# Load model function
@st.cache_resource
def load_model():
    """Load the pretrained ConvNeXt model."""
    model = models.convnext_tiny(weights=None)

    # Freeze base
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, 10)

    # Load checkpoint
    checkpoint = torch.load('saved_models/baseline_convnext/checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.model.eval()
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            if output.requires_grad:
                output.register_hook(self._save_gradient)

        self.target_layer.register_forward_hook(forward_hook)

    def _save_gradient(self, grad):
        self.gradients = grad

    def generate(self, input_tensor, class_idx):
        input_tensor = input_tensor.clone().detach().to(device).requires_grad_(True)
        self.model.zero_grad()
        self.gradients = None

        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward()

        if self.gradients is None:
            raise RuntimeError("Gradients were not captured.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu()


# Preprocessing function
def preprocess_image(image):
    """Preprocess uploaded image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Convert to grayscale if RGB
    if image.mode == 'RGB':
        image = image.convert('L')

    return transform(image).unsqueeze(0)


# Prediction function
def predict_with_gradcam(model, image_tensor, grad_cam):
    """Make prediction and generate Grad-CAM visualization."""
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Generate Grad-CAM
    cam = grad_cam.generate(image_tensor, predicted_class.item())
    cam = cam.unsqueeze(1)
    cam_resized = F.interpolate(
        cam,
        size=image_tensor.shape[2:],
        mode="bilinear",
        align_corners=False
    ).squeeze().numpy()

    return predicted_class.item(), confidence.item(), probabilities[0], cam_resized


# Main app
def main():
    st.title("👕 Fashion-MNIST Classifier with Grad-CAM")
    st.markdown("""
    Upload a grayscale fashion item image (28×28 recommended) and the model will:
    - Classify it into one of 10 fashion categories
    - Show prediction confidence
    - Generate a Grad-CAM heatmap showing which regions influenced the decision
    """)

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("""
    **Model**: ConvNeXt (tiny) pretrained on ImageNet

    **Dataset**: Fashion-MNIST (10 classes)

    **Method**: Transfer Learning (Feature Extraction)

    **Validation Accuracy**: ~90%
    """)

    st.sidebar.header("Sample Classes")
    st.sidebar.text("\n".join([f"{i}: {name}" for i, name in enumerate(class_names)]))

    # Load model
    try:
        model = load_model()
        grad_cam = GradCAM(model, model.features[-1][-1])
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a fashion item image",
        type=["png", "jpg", "jpeg"],
        help="Upload a grayscale image of a clothing item"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Uploaded Image")
            st.image(image, caption="Original Image", use_container_width=True)

        # Preprocess and predict
        with st.spinner("🔍 Analyzing image..."):
            image_tensor = preprocess_image(image)
            pred_class, confidence, probabilities, cam = predict_with_gradcam(
                model, image_tensor, grad_cam
            )

        with col2:
            st.subheader("🎯 Prediction Results")
            st.metric("Predicted Class", class_names[pred_class])
            st.metric("Confidence", f"{confidence * 100:.2f}%")

            # Top 3 predictions
            st.markdown("**Top 3 Predictions:**")
            top3_indices = torch.topk(probabilities, 3).indices
            for i, idx in enumerate(top3_indices):
                st.write(f"{i + 1}. {class_names[idx]}: {probabilities[idx] * 100:.2f}%")

        # Grad-CAM visualization
        st.subheader("🔥 Grad-CAM Heatmap")
        st.markdown("**Red regions** indicate areas that most influenced the model's decision.")

        # Create Grad-CAM overlay
        fig, ax = plt.subplots(figsize=(6, 6))

        # Prepare image for display
        img_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        ax.imshow(img_np)
        ax.imshow(cam, cmap="jet", alpha=0.5)
        ax.set_title(f"Grad-CAM for '{class_names[pred_class]}'", fontsize=14, fontweight='bold')
        ax.axis('off')

        st.pyplot(fig)

        # Probability distribution
        st.subheader("📊 Class Probability Distribution")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(class_names, probabilities.cpu().numpy(), color='steelblue', edgecolor='black')
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_title('Prediction Confidence Across All Classes', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)


if __name__ == "__main__":
    main()
