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


# Scratch CNN Architecture
class ScratchCNN(nn.Module):
    def __init__(self):
        super(ScratchCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Changed: 32 -> 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Changed: 64 -> 32
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Changed: 64*7*7=3136 -> 32*7*7=1568
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # Changed: 64*7*7 -> 32*7*7
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Load ConvNeXt model
@st.cache_resource
def load_convnext_model():
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


# Load Scratch CNN model
@st.cache_resource
def load_scratch_model():
    """Load the scratch CNN model."""
    model = ScratchCNN()

    # Load checkpoint
    checkpoint = torch.load('saved_models/scratch_cnn/checkpoint.pth', map_location=device)
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
def preprocess_image(image, model_type):
    """Preprocess uploaded image based on model type."""
    if model_type == "ConvNeXt":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:  # Scratch CNN
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
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

    # Model selection in sidebar
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose Model Architecture",
        ["ConvNeXt (Transfer Learning)", "Scratch CNN"]
    )

    model_type = "ConvNeXt" if "ConvNeXt" in model_choice else "Scratch"

    # Model-specific information
    st.sidebar.header("About")
    if model_type == "ConvNeXt":
        st.sidebar.info("""
        **Model**: ConvNeXt (tiny) pretrained on ImageNet

        **Dataset**: Fashion-MNIST (10 classes)

        **Method**: Transfer Learning (Feature Extraction)

        **Input Size**: 224×224 RGB

        **Parameters**: ~28M (frozen) + classifier

        **Validation Accuracy**: ~90%
        """)
    else:
        st.sidebar.info("""
        **Model**: Custom CNN (trained from scratch)

        **Dataset**: Fashion-MNIST (10 classes)

        **Method**: End-to-end training

        **Input Size**: 28×28 Grayscale

        **Parameters**: ~250K (all trainable)

        **Validation Accuracy**: ~85%
        """)

    st.sidebar.header("Sample Classes")
    st.sidebar.text("\n".join([f"{i}: {name}" for i, name in enumerate(class_names)]))

    # Main description
    st.markdown(f"""
    Upload a grayscale fashion item image and the **{model_choice}** will:
    - Classify it into one of 10 fashion categories
    - Show prediction confidence
    - Generate a Grad-CAM heatmap showing which regions influenced the decision

    **Currently using**: {model_choice}
    """)

    # Load selected model
    try:
        if model_type == "ConvNeXt":
            model = load_convnext_model()
            grad_cam = GradCAM(model, model.features[-1][-1])
        else:
            model = load_scratch_model()
            grad_cam = GradCAM(model, model.conv2)

        st.success(f" {model_choice} loaded successfully!")

        # Display model info
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Device", str(device).upper())
        with col_info2:
            input_size = "224×224 RGB" if model_type == "ConvNeXt" else "28×28 Grayscale"
            st.metric("Input Size", input_size)
        with col_info3:
            params = sum(p.numel() for p in model.parameters())
            st.metric("Parameters", f"{params:,}")

    except Exception as e:
        st.error(f" Error loading model: {e}")
        st.error("Make sure the checkpoint file exists in the correct path:")
        if model_type == "ConvNeXt":
            st.code("saved_models/baseline_convnext/checkpoint.pth")
        else:
            st.code("saved_models/scratch_cnn/checkpoint.pth")
        st.stop()

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a fashion item image",
        type=["png", "jpg", "jpeg"],
        help="Upload a grayscale or color image of a clothing item"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Uploaded Image")
            st.image(image, caption="Original Image", use_container_width=True)
            st.caption(f"Image size: {image.size[0]}×{image.size[1]} pixels")
            st.caption(f"Image mode: {image.mode}")

        # Preprocess and predict
        with st.spinner(f"🔍 Analyzing image with {model_choice}..."):
            image_tensor = preprocess_image(image, model_type)
            pred_class, confidence, probabilities, cam = predict_with_gradcam(
                model, image_tensor, grad_cam
            )

        with col2:
            st.subheader("🎯 Prediction Results")
            st.metric("Predicted Class", class_names[pred_class])
            st.metric("Confidence", f"{confidence * 100:.2f}%")

            # Confidence indicator
            if confidence > 0.8:
                st.success("High confidence prediction")
            elif confidence > 0.5:
                st.warning("Medium confidence prediction")
            else:
                st.error("Low confidence prediction")

            # Top 3 predictions
            st.markdown("**Top 3 Predictions:**")
            top3_indices = torch.topk(probabilities, 3).indices
            for i, idx in enumerate(top3_indices):
                prob = probabilities[idx] * 100
                st.write(f"{i + 1}. **{class_names[idx]}**: {prob:.2f}%")
                st.progress(int(prob))

        st.markdown("---")

        # Grad-CAM visualization
        st.subheader("🔥 Grad-CAM Heatmap")
        st.markdown("**Red regions** indicate areas that most influenced the model's decision.")

        col_viz1, col_viz2, col_viz3 = st.columns(3)

        with col_viz1:
            st.markdown("**Original Image**")
            # Prepare image for display
            img_np = image_tensor.squeeze()
            if model_type == "ConvNeXt":
                img_np = img_np.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = img_np.cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

            fig1, ax1 = plt.subplots(figsize=(5, 5))
            if model_type == "ConvNeXt":
                ax1.imshow(img_np)
            else:
                ax1.imshow(img_np, cmap='gray')
            ax1.axis('off')
            st.pyplot(fig1)

        with col_viz2:
            st.markdown("**Grad-CAM Heatmap**")
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            ax2.imshow(cam, cmap="jet")
            ax2.axis('off')
            st.pyplot(fig2)

        with col_viz3:
            st.markdown("**Overlay**")
            fig3, ax3 = plt.subplots(figsize=(5, 5))
            if model_type == "ConvNeXt":
                ax3.imshow(img_np)
            else:
                ax3.imshow(img_np, cmap='gray')
            ax3.imshow(cam, cmap="jet", alpha=0.5)
            ax3.set_title(f"Grad-CAM for '{class_names[pred_class]}'", fontsize=12, fontweight='bold')
            ax3.axis('off')
            st.pyplot(fig3)

        st.markdown("---")

        # Probability distribution
        st.subheader("📊 Class Probability Distribution")
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        bars = ax4.bar(class_names, probabilities.cpu().numpy(), color='steelblue', edgecolor='black')

        # Highlight predicted class
        bars[pred_class].set_color('darkred')

        ax4.set_ylabel('Probability', fontsize=12)
        ax4.set_xlabel('Class', fontsize=12)
        ax4.set_title(f'Prediction Confidence Across All Classes (Model: {model_choice})',
                      fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_ylim(0, 1)
        plt.tight_layout()
        st.pyplot(fig4)

        # Model comparison suggestion
        st.info(f" **Tip**: Try switching to the other model to compare predictions! "
                f"Currently using: **{model_choice}**")


if __name__ == "__main__":
    main()
