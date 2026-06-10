import torch
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None

        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x):
        # Forward
        output = self.model(x)
        pred_class = output.argmax(dim=1)

        # Backward
        self.model.zero_grad()
        output[0, pred_class.item()].backward()

        # Gradients & activations
        gradients = self.gradients[0]        # [C, H, W]
        activations = self.activations[0]    # [C, H, W]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1,2))

        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:], device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Normalize
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam.detach().cpu().numpy()