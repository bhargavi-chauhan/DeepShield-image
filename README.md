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
│             ├── fake
│             ├── real
│       ├── val (fake & real)
│       └── test (fake & real)
│
├── models/
│   ├── image_model.py
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
│   ├── gradcam_compare.py
│   └── explainability.py
│
├── training_model/
|   ├── train_image.py
|
├── inference_model/
|   ├── infer_image.py
|   └── onnx_inference.py
|
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── requirements.txt
```

# Ways to run DeepShield: 

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
python -m training_model.train_image
```

Run prediction on a test image:
```
python -m inference_model.infer_image --image test_dataset/images/<test_image#>.<img_format>
```
or

```
python -m inference_model.onnx_inference --image test_dataset/images/<test_image#>.<img_format>
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
python -m training_model.train_image
```
Run prediction on a test image:
```
docker run -it deepshield python -m inference_model.infer_image --image test_dataset/images/<test_image#>.<img_format>
```
or
```
docker run -it deepshield python -m inference_model.onnx_inference --image test_dataset/images/<test_image#>.<img_format>
```






