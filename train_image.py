import warnings
warnings.filterwarnings("ignore")

import os
import torch
torch.set_num_threads(8)
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report
)
import pandas as pd

from models.image_model import DeepShieldImageModel
from utils.preprocess import DeepfakeImageDataset
# from utils.gradcam import generate_gradcam
from utils.gradcam_compare import generate_fake_vs_real_gradcam
 
# =============================
# Device
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =============================
# Directories
# =============================
os.makedirs("models", exist_ok=True)

os.makedirs("outputs/val_metrics", exist_ok=True)
os.makedirs("outputs/test_metrics", exist_ok=True)
os.makedirs("outputs/explainability_result", exist_ok=True)

def main(): 

    # =============================
    # Load Dataset
    # =============================
    train_dataset = DeepfakeImageDataset("datasets/images/train", train=True)
    val_dataset   = DeepfakeImageDataset("datasets/images/val", train=False)
    test_dataset  = DeepfakeImageDataset("datasets/images/test", train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    print("Training samples   :", len(train_dataset))
    print("Validation samples :", len(val_dataset))
    print("Test samples       :", len(test_dataset))

    # =============================
    # Model
    # =============================
    model = DeepShieldImageModel().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # =============================
    # Training Config
    # =============================
    epochs = 30
    patience = 5
    best_val_loss = float("inf")
    early_stop_counter = 0


    # =============================
    # Evaluation Function
    # =============================
    def evaluate_model(model, loader, save_prefix="val"):

        model.eval()
        total_loss = 0

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for imgs, labels in loader:

                imgs = imgs.to(device)
                labels_gpu = labels.unsqueeze(1).float().to(device)

                logits = model(imgs)

                loss = criterion(logits, labels_gpu)
                total_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.numpy().flatten())

        avg_loss = total_loss / len(loader)

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0

        print(f"\n📊 {save_prefix.upper()} Metrics")
        print("Loss     :", round(avg_loss, 4))
        print("Accuracy :", round(acc, 4))
        print("Precision:", round(prec, 4))
        print("Recall   :", round(rec, 4))
        print("F1 Score :", round(f1, 4))
        print("AUC      :", round(auc, 4))

        # =============================
        # Select folder
        # =============================
        if save_prefix == "val":
            save_dir = "outputs/val_metrics"
        else:
            save_dir = "outputs/test_metrics"

        # =============================
        # Confusion Matrix
        # =============================
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure()
        plt.imshow(cm)
        plt.title(f"{save_prefix.upper()} Confusion Matrix")
        plt.colorbar()

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.xticks([0,1], ["Fake","Real"])
        plt.yticks([0,1], ["Fake","Real"])

        plt.tight_layout()

        plt.savefig(f"{save_dir}/{save_prefix}_confusion_matrix.png")
        plt.close()

        # =============================
        # ROC Curve
        # =============================
        fpr, tpr, _ = roc_curve(all_labels, all_probs)

        plt.figure()
        plt.plot(fpr, tpr)

        plt.title(f"{save_prefix.upper()} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.tight_layout()

        plt.savefig(f"{save_dir}/{save_prefix}_roc_curve.png")
        plt.close()

        # =============================
        # Classification Report
        # =============================
        report = classification_report(
            all_labels,
            all_preds,
            target_names=["Fake", "Real"],
            zero_division=0
        )

        with open(f"{save_dir}/{save_prefix}_classification_report.txt", "w") as f:
            f.write(report)

        # =============================
        # Metrics CSV
        # =============================
        metrics_df = pd.DataFrame([{
            "Loss": avg_loss,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "AUC": auc
        }])

        metrics_df.to_csv(f"{save_dir}/{save_prefix}_metrics.csv", index=False)

        return avg_loss


    # =============================
    # Training Loop
    # =============================
    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
    
            imgs = imgs.to(device)
            labels = labels.unsqueeze(1).float().to(device)

            logits = model(imgs)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}")

        # =============================
        # Validation
        # =============================
        val_loss = evaluate_model(model, val_loader, save_prefix="val")

        # =============================
        # Early Stopping
        # =============================
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            early_stop_counter = 0

            torch.save(model.state_dict(), "models/best_image_model.pth")

            print("✅ Best model saved!")

        else:

            early_stop_counter += 1

            print(f"Early stop counter: {early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                print("⛔ Early stopping triggered!")
                break


    print("\n🎯 Training Complete!")

    # =============================
    # Final Test Evaluation
    # =============================
    model.load_state_dict(torch.load("models/best_image_model.pth"))

    model.to(device)

    evaluate_model(model, test_loader, save_prefix="test")

    torch.save(model.state_dict(), "models/final_image_model.pth")

    print("Final model saved.")

    # =============================
    # # Grad-CAM (Explainability)
    # # =============================
    # sample_batch, _ = next(iter(test_loader))

    # sample_img = sample_batch[0].to(device)
    # sample_img = sample_img.unsqueeze(0)

    # generate_gradcam(
    #     model,
    #     sample_img,
    #     device,
    #     save_path="outputs/explainability_result/training_gradcam.png"
    # )
    # =============================
    # Fake vs Real GradCAM
    # =============================

    fake_img = None
    real_img = None

    for imgs, labels in test_loader:

        for i in range(len(labels)):

            if labels[i] == 0 and fake_img is None:
                fake_img = imgs[i]

            if labels[i] == 1 and real_img is None:
                real_img = imgs[i]

            if fake_img is not None and real_img is not None:
                break

        if fake_img is not None and real_img is not None:
            break


    generate_fake_vs_real_gradcam(
        model,
        fake_img.to(device),
        real_img.to(device),
        device,
        save_path="outputs/explainability_result/fake_vs_real_training_gradcam.png"
    )
    # =============================
    # Export to ONNX
    # =============================
    model.eval()

    dummy_input = torch.randn(1,3,224,224).to(device)

    onnx_path = "models/deepshield_efficientnet.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":{0:"batch_size"},
            "logits":{0:"batch_size"}
        },
        opset_version=18
    )

    print(f"✅ ONNX model exported to {onnx_path}")

if __name__ == "__main__":
    main()