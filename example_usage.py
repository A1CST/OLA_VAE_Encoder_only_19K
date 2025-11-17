# example_usage.py
import torch
from ovae import OrganicEncoder
from PIL import Image
import torchvision.transforms as T

# Load encoder (auto-loads encoder_weights.json on CPU)
encoder = OrganicEncoder().eval()

# Basic image preprocessing (adjust to your pipeline)
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

img = Image.open("example.jpg").convert("RGB")
tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 256, 256]

with torch.no_grad():
    latent = encoder(tensor)  # shape: [1, 4]

print(latent)
