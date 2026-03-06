import torch
import numpy as np

# -------------------------------
# Confidence Zone Classification
# -------------------------------
def confidence_zone(prob):
    if prob >= 0.8 or prob <= 0.2:
        return "HIGH"
    elif prob >= 0.6 or prob <= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

# ------------------------------------
# Authenticity Prediction (Upgraded)
# ------------------------------------
def predict_authenticity(model, image_tensor, device):
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor.to(device))
        prob_real = torch.sigmoid(logits).item()

    # 🔥 Innovation 1: Reject Option
    if 0.45 <= prob_real <= 0.55:
        return {
            "score": prob_real,
            "label": "UNCERTAIN",
            "confidence": "LOW",
            "action": "REQUIRES HUMAN REVIEW"
        }

    label = "REAL" if prob_real >= 0.5 else "FAKE"
    confidence = confidence_zone(prob_real)

    return {
        "score": prob_real,
        "label": label,
        "confidence": confidence,
        "action": "AUTO"
    }


