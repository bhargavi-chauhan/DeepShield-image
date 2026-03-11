import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DeepfakeImageDataset(Dataset):

    def __init__(self, root_dir, transform=None, train=True):

        self.root_dir = root_dir
        self.classes = ["fake", "real"]
        self.class_to_idx = {"fake": 0, "real": 1}
        self.samples = []

        if not os.path.exists(root_dir):
            raise ValueError(f"Dataset directory not found: {root_dir}")

        for label in self.classes:

            folder = os.path.join(root_dir, label)

            if not os.path.exists(folder):
                raise ValueError(f"Missing class folder: {folder}")

            # Faster directory scanning
            for entry in os.scandir(folder):

                if entry.is_file() and entry.name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp")
                ):
                    self.samples.append(
                        (entry.path, self.class_to_idx[label])
                    )

        # Shuffle dataset order
        random.shuffle(self.samples)

        print(f"Loaded {len(self.samples)} images from {root_dir}")

        # Transforms
        if transform is not None:
            self.transform = transform

        else:

            if train:

                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]
                    )
                ])

            else:

                self.transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]
                    )
                ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        image = self.transform(image)

        return image, label
