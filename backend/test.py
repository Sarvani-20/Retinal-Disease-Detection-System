from PIL import Image
import torch
from torchvision import transforms

# device (optional)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image path
image = Image.open(r"C:\EyeDiseases\CombinedDataset\sever_gla\AIMER Glaucoma_1705_left.jpeg").convert("RGB")

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# tensor
tensor = transform(image).unsqueeze(0).to(DEVICE)
print("Tensor shape:", tensor.shape)