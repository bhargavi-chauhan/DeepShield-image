import torch
import numpy as np
from torchvision import transforms

def multi_crop_inference(model, image, device, transform, runs=5):
    scores = []

    model.eval()
    for _ in range(runs):
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.sigmoid(logits).item()
            scores.append(prob)

    return float(np.mean(scores))
