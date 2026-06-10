import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path, device):
    model = models.efficientnet_b0(weights=None)

    num_ftrs = model.classifier[1].in_features

    # ✅ EXACT SAME AS TRAINING
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 9)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model
def clean_label(label):
    mapping = {
        'DRY AMD': 'ARMD Dry',
        'WET AMD': 'ARMD Wet',
        'MILDNPDR': 'DR Mild',
        'MODERATE': 'DR Moderate',
        'SEVERNPDR': 'DR Severe',
        'mild-gla': 'Glaucoma Mild',
        'moderate_gla': 'Glaucoma Moderate',
        'sever_gla': 'Glaucoma Severe',
        'Healthy': 'Healthy'
    }
    return mapping.get(label, label)