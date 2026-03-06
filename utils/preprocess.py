import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DeepfakeImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        """
        Args:
            root_dir (str): Path to dataset folder (containing 'fake' and 'real')
            transform (torchvision.transforms, optional): Custom transforms
            train (bool): Whether dataset is for training (enables augmentation)
        """

        self.root_dir = root_dir
        self.classes = ["fake", "real"]
        self.class_to_idx = {"fake": 0, "real": 1}
        self.samples = []

        # Validate directory
        if not os.path.exists(root_dir):
            raise ValueError(f"Dataset directory not found: {root_dir}")

        # Collect image paths
        for label in self.classes:
            folder = os.path.join(root_dir, label)

            if not os.path.exists(folder):
                raise ValueError(f"Missing class folder: {folder}")

            for file in os.listdir(folder):
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append(
                        (os.path.join(folder, file), self.class_to_idx[label])
                    )

        # ==============================
        # Transforms
        # ==============================

        if transform is not None:
            self.transform = transform
        else:
            if train:
                # Data Augmentation for training
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:
                # No augmentation for validation/testing
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image: {path}") from e

        image = self.transform(image)

        return image, label