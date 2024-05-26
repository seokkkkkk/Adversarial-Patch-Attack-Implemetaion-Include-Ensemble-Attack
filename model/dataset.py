from torch.utils.data import Dataset
import cv2 as cv
import torch

class ImageDataset(Dataset):
    def __init__(self, image_paths, device, img_size=(224, 224)):
        self.image_paths = image_paths
        self.device = device
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        # bgr to rgb
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image = image / 255.0

        if image is None:
            raise ValueError(f"Failed to load image at path: {image_path}")

        if image.ndim == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        image = cv.resize(image, self.img_size)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        return image
