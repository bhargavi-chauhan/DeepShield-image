import argparse
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# =============================
# Parse Image Argument
# =============================

parser = argparse.ArgumentParser(description="DeepShield ONNX Inference")
parser.add_argument("--image", type=str, required=True, help="Path to image file")

args = parser.parse_args()
img_path = args.image

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

# =============================
# Load ONNX Model
# =============================

onnx_model_path = "models/deepshield_efficientnet.onnx"

session = ort.InferenceSession(onnx_model_path)

print("ONNX model loaded successfully!")

# =============================
# Image Preprocessing
# =============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(img_path).convert("RGB")

tensor = transform(image)

# Convert to numpy
input_tensor = tensor.unsqueeze(0).numpy()

# =============================
# Run Inference
# =============================

outputs = session.run(
    None,
    {"input": input_tensor}
)

logits = outputs[0]

# Apply sigmoid
prob = 1 / (1 + np.exp(-logits))

score = prob[0][0]

# =============================
# Prediction
# =============================

if score >= 0.5:
    label = "Real"
else:
    label = "Fake"

confidence = abs(score - 0.5) * 2

print("\n🔍 DeepShield ONNX Prediction")
print("--------------------------------")
print(f"Image Path : {img_path}")
print(f"Score      : {score:.4f}")
print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2f}")
