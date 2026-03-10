import os
import argparse
import torch
from PIL import Image
from torchvision import transforms

from models.image_model import DeepShieldImageModel
from utils.inference import predict_authenticity
from utils.explainability import GradCAM, explanation_strength, overlay_heatmap
from utils.multicrop import multi_crop_inference

# ------------------------------------------------
# Parse Input Argument
# ------------------------------------------------
parser = argparse.ArgumentParser(description="DeepShield Image Inference")
parser.add_argument("--image", type=str, required=True, help="Path to image file")

args = parser.parse_args()
img_path = args.image

# ------------------------------------------------
# Device
# ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------
# Ensure output folder exists
# ------------------------------------------------
os.makedirs("outputs/explainability_result", exist_ok=True)

# ------------------------------------------------
# Load Model
# ------------------------------------------------
model = DeepShieldImageModel().to(device)

model.load_state_dict(
    torch.load("models/best_image_model.pth", map_location=device)
)

model.eval()

print("✅ Model loaded successfully")

# ------------------------------------------------
# Image Transform
# ------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------------------------------------
# Load Image
# ------------------------------------------------
if not os.path.exists(img_path):
    raise FileNotFoundError(f"❌ Image not found: {img_path}")

image = Image.open(img_path).convert("RGB")

image_tensor = transform(image).unsqueeze(0).to(device)

# ------------------------------------------------
# 🔥 Multi-Crop Consensus Inference
# ------------------------------------------------
print("\nRunning Multi-Crop Consensus...")

avg_score = multi_crop_inference(
    model,
    image,
    device,
    transform
)

# ------------------------------------------------
# 🔥 Base Prediction
# ------------------------------------------------
result = predict_authenticity(
    model,
    image_tensor,
    device
)

# ------------------------------------------------
# 🔥 Final Decision
# ------------------------------------------------
if avg_score < 0.5:
    final_label = "FAKE"
else:
    final_label = "REAL"

confidence = result["confidence"]
action = result["action"]

# ------------------------------------------------
# Prediction Report
# ------------------------------------------------
print("\n🔍 DeepShield Prediction Report")
print("--------------------------------")

print(f"Image Path         : {img_path}")
print(f"Authenticity Score : {avg_score:.4f}")
print(f"Prediction         : {final_label}")
print(f"Confidence Level   : {confidence}")
print(f"System Action      : {action}")

# ------------------------------------------------
# 🔥 Explainability (Grad-CAM)
# ------------------------------------------------
print("\nGenerating Grad-CAM...")

gradcam = GradCAM(model, model.backbone.features[-1])

heatmap = gradcam.generate(image_tensor)

strength = explanation_strength(heatmap)

if strength < 0.15:
    print("⚠ WARNING: Weak visual evidence detected")
    print("⚠ Prediction reliability is LOW")

output_path = "outputs/explainability_result/inference_gradcam.jpg"

overlay_heatmap(
    heatmap,
    img_path,
    output_path
)

print(f"✅ Grad-CAM saved to {output_path}")

print("\n✅ DeepShield analysis completed.")
