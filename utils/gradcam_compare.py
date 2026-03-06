import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_fake_vs_real_gradcam(model, fake_img, real_img, device, save_path):

    model.eval()

    target_layer = model.backbone.features[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])

    fake_tensor = fake_img.unsqueeze(0).to(device)
    real_tensor = real_img.unsqueeze(0).to(device)

    # Generate CAMs
    fake_cam = cam(input_tensor=fake_tensor)[0]
    real_cam = cam(input_tensor=real_tensor)[0]

    fake_np = fake_img.permute(1,2,0).cpu().numpy()
    real_np = real_img.permute(1,2,0).cpu().numpy()

    fake_np = (fake_np - fake_np.min()) / (fake_np.max() - fake_np.min())
    real_np = (real_np - real_np.min()) / (real_np.max() - real_np.min())

    fake_heatmap = show_cam_on_image(fake_np, fake_cam, use_rgb=True)
    real_heatmap = show_cam_on_image(real_np, real_cam, use_rgb=True)

    # Plot comparison
    plt.figure(figsize=(10,6))

    plt.subplot(2,2,1)
    plt.imshow(real_np)
    plt.title("Real Image")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.imshow(real_heatmap)
    plt.title("Real GradCAM")
    plt.axis("off")

    plt.subplot(2,2,3)
    plt.imshow(fake_np)
    plt.title("Fake Image")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.imshow(fake_heatmap)
    plt.title("Fake GradCAM")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("✅ Fake vs Real Training GradCAM saved to:", save_path)