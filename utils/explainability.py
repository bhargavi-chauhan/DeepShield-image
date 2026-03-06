import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.zero_grad()
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        target = output if output.item() >= 0.5 else 1 - output
        target.backward()

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(activations.shape[1]):
            activations[:, i] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= (torch.max(heatmap) + 1e-8)

        return heatmap.cpu().numpy()

# -------------------------------
# 🔥 Explanation Strength Metric
# -------------------------------
def explanation_strength(heatmap):
    return np.mean(heatmap)

# -------------------------------
# Heatmap Overlay
# -------------------------------
def overlay_heatmap(heatmap, original_image_path, output_path):
    image = cv2.imread(original_image_path)
    image = cv2.resize(image, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(image, 0.4, heatmap, 0.6, 0)
    cv2.imwrite(output_path, superimposed)

    print(f"Grad-CAM saved at {output_path}")
