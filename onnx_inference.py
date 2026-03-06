import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

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

img_path = "test_dataset/images/test_image1.jpg"

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
print(f"Score      : {score:.4f}")
print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2f}")