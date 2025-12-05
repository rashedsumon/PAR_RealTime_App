

## **5️⃣ data_loader.py** – Download & preprocess dataset


import kagglehub
import os
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

# Download dataset automatically
def download_dataset():
    path = kagglehub.dataset_download("mahdiehhajian/clothing-attributes-dataset")
    print("Path to dataset files:", path)
    return path

# Custom dataset class
class ClothingDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.images = list(self.image_dir.glob("*.jpg"))  # adjust if images in subfolders

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Placeholder labels (replace with actual attribute parsing)
        labels = {
            "gender": 0,
            "age": 0,
            "hair_length": 0,
            "upper_clothing": 0,
            "lower_clothing": 0,
            "accessories": [0,0,0]  # multi-label
        }
        return image, labels

# Define transforms
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
