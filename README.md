# DeepShield-image
Deep learning system with Explainable AI Framework for Deepfake Detection in images and Media Authenticity.

## 📌 Key Features
✔ Deepfake Image Detection using EfficientNet-B0 CNN backbone

✔ Grad-CAM Explainability 

✔ Fake vs Real Heatmap Comparison

✔ Multi-Crop Consensus Inference 

✔ ONNX Model Export for fast deployment

✔ Evaluation Metrics & Visualization

✔ Docker + NVIDIA GPU Support

## 📂 Project Structure
```
DeepShield
│
├── datasets/
│   └── images/
│       ├── train
│       ├── val
│       └── test
│
├── models/
│   ├── best_image_model.pth
│   ├── final_image_model.pth
│   └── deepshield_efficientnet.onnx
│
├── outputs/
│   ├── val_metrics/
│   ├── test_metrics/
│   └── explainability_result/
│
├── utils/
│   ├── preprocess.py
│   ├── inference.py
│   ├── multicrop.py
│   ├── gradcam.py
│   ├── gradcam_compare.py
│   └── explainability.py
│
├── train_image.py
├── infer_image.py
├── onnx_inference.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

# ways to run DeepShield: 

### 🚀 Local Python

Clone the repository:
```
git clone https://github.com/bhargavi-chauhan/DeepShield-image
cd DeepShield
```

Install dependencies:
```
pip install -r requirements.txt
```

Training the Model:
```
python train_image.py
```

Run prediction on a test image:
```
python infer_image.py
```
or

```
python onnx_inference.py
```

### 🐳 Docker Support

DeepShield supports containerized execution using Docker.

Build Docker image:
```
docker build -t deepshield .
```
Run container with GPU:
```
docker run --gpus all -it deepshield
```
Inside container:
```
python train_image.py
```
Run prediction on a test image:
```
python infer_image.py
```
or
```
python onnx_inference.py
```






