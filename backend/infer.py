import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from gradcam import GradCAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = [
    'DRY AMD', 'Healthy', 'MILDNPDR', 'MODERATE',
    'SEVERNPDR', 'WET AMD', 'mild-gla',
    'moderate_gla', 'sever_gla'
]

# -------- MODEL --------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, len(classes))
)
model.load_state_dict(torch.load(
    r"C:\EyeProject\model\best_combined_model.pth",
    map_location=DEVICE
))
model = model.to(DEVICE)
model.eval()

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -------- PREDICT --------
def predict(image_path, uid):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0][pred].item()
    label = classes[pred] if pred < len(classes) else "Unknown"

    # Save original
    original_path = f"outputs/{uid}_original.jpg"
    image.save(original_path)

    # Grad-CAM
    target_layer = model.features[-1][0]
    cam = GradCAM(model, target_layer, DEVICE).generate(tensor)

    cam = cv2.resize(cam, (224,224))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    original = cv2.resize(np.array(image), (224,224))
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    heatmap_path = f"gradcam_outputs/{uid}_gradcam.jpg"
    cv2.imwrite(heatmap_path, overlay)

    return (
        label,
        conf,
        f"http://localhost:8000/outputs/{uid}_original.jpg",
        f"http://localhost:8000/gradcam/{uid}_gradcam.jpg"
    )