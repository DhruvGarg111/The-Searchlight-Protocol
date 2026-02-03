import os
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INCREASE_CONTRAST = 1.8

class DroneImageLoader:
    def __init__(self, max_dim):
        self.max_dim = max_dim
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        img = Image.open(image_path).convert('RGB')
        original_size = img.size

        if INCREASE_CONTRAST != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(INCREASE_CONTRAST)
            print(f"  Applied contrast enhancement: x{INCREASE_CONTRAST}")

        original_np = np.array(img)

        width, height = original_size
        scale = min(self.max_dim / width, self.max_dim / height, 1.0)

        if scale < 1.0:
            new_size = (int(width * scale), int(height * scale))
            img_resized = img.resize(new_size, Image.LANCZOS)
            print(f"  Original: {original_size} -> Downsampled for CAM: {new_size} (scale: {scale:.3f})")
        else:
            img_resized = img
            scale = 1.0
            print(f"  Loaded at original resolution: {original_size} (no downsampling needed)")

        tensor = self.transform(img_resized).unsqueeze(0).to(device)
        print(f"  Tensor shape: {tensor.shape}, Memory: {tensor.element_size() * tensor.nelement() / 1024**2:.2f} MB")

        return original_np, tensor, original_size, scale
