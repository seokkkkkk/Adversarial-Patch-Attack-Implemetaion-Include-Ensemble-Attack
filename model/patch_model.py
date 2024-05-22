import os
import torch
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt
from random import sample
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, image_paths, resize_to=(416, 416)):
        self.image_paths = image_paths
        self.resize_to = resize_to
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize_to),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv.imread(self.image_paths[idx], cv.IMREAD_COLOR)  # BGR로 읽음
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = self.transform(image)
        return image


def load_model():
    print("모델 로드")
    model = YOLO('yolov8n.pt', verbose=False).to(device)
    print("모델 로드 완료")
    return model


def patch_init(patch_image, patch_size, patch_shape):
    print("패치 초기화")
    image = cv.imread(patch_image, cv.IMREAD_UNCHANGED)
    if image is None:
        print("이미지 로드 실패. 랜덤 이미지 생성")
        image = np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)
    else:
        image = cv.resize(image, (patch_size, patch_size))

    if image.shape[2] == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)

    shape = np.zeros((patch_size, patch_size), dtype=np.uint8)
    if patch_shape == "circle":
        center = (patch_size // 2, patch_size // 2)
        radius = patch_size // 2
        cv.circle(shape, center, radius, 255, -1)
    elif patch_shape == "rectangle":
        cv.rectangle(shape, (0, 0), (patch_size, patch_size), 255, -1)
    else:
        print("유효하지 않은 patch_shape")
        return None

    alpha_channel = np.where(shape == 255, 255, 0).astype(np.uint8)
    image[:, :, 3] = alpha_channel

    patch = torch.from_numpy(image).float().to(device).permute(2, 0, 1) / 255.0
    patch.requires_grad_(True)
    print("패치 초기화 완료")

    patch_save(patch, "patch/", "initial_patch")

    return patch


def patch_to_image(image, patch, x, y):
    patch_height, patch_width = patch.shape[1:]

    patch_np = patch.permute(1, 2, 0).cpu().detach().numpy()
    patch_np = (patch_np * 255).astype(np.uint8)

    alpha_s = patch_np[:, :, 3] / 255.0
    patch_np = patch_np[:, :, :3]

    image_np = image.permute(1, 2, 0).cpu().detach().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        image_np[y:y + patch_height, x:x + patch_width, c] = (
                alpha_s * patch_np[:, :, c] + alpha_l * image_np[y:y + patch_height, x:x + patch_width, c]
        )

    image_np = image_np.astype(np.float32) / 255.0
    image = torch.tensor(image_np, dtype=torch.float32, device=device).permute(2, 0, 1)

    return image


def patch_transform(patch, angle, scale, device='cpu'):
    patch_np = patch.permute(1, 2, 0).cpu().detach().numpy()
    patch_np = (patch_np * 255).astype(np.uint8)

    original_height, original_width = patch_np.shape[:2]

    resized_width = int(original_width * scale)
    resized_height = int(original_height * scale)
    resized_patch_np = cv.resize(patch_np, (resized_width, resized_height), interpolation=cv.INTER_AREA)

    new_center = (resized_width // 2, resized_height // 2)

    matrix = cv.getRotationMatrix2D(new_center, angle, 1.0)
    rotated_patch_np = cv.warpAffine(resized_patch_np, matrix, (resized_width, resized_height))

    patch_transformed_np = rotated_patch_np.astype(np.float32) / 255.0
    patch_transformed = torch.tensor(patch_transformed_np, dtype=torch.float32, device=device).permute(2, 0, 1)
    patch_transformed.requires_grad_(True)

    return patch_transformed


def patch_save(patch, path, name):
    print("최소 손실 패치 저장: " + name)
    patch_np = patch.permute(1, 2, 0).cpu().detach().numpy() * 255.0
    patch_np = patch_np.astype(np.uint8)

    os.makedirs(path, exist_ok=True)
    cv.imwrite(os.path.join(path, f"{name}.png"), patch_np)
    print(f"{os.path.join(path, f'{name}.png')}에 저장 완료")


def compute_loss(pred, target_class):
    # 여러 예측을 고려한 손실 계산
    loss = 0
    for detection in pred:
        if detection[5] == target_class:  # detection[5]는 예측된 클래스
            loss -= torch.log(detection[4])  # detection[4]는 해당 클래스의 확률
    return loss


def patch_optimize(save_path="patch/"):
    print("패치 최적화 시작")

    train_image_paths = [os.path.join(train_image_path, f) for f in os.listdir(train_image_path)]
    val_image_paths = [os.path.join(val_image_path, f) for f in os.listdir(val_image_path)]

    if len(train_image_paths) > 5000:
        train_image_paths = sample(train_image_paths, 5000)
    if len(val_image_paths) > 2000:
        val_image_paths = sample(val_image_paths, 2000)

    train_dataset = ImageDataset(train_image_paths)
    val_dataset = ImageDataset(val_image_paths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Train Images: {len(train_dataset)}")
    print(f"Validation Images: {len(val_dataset)}")

    optimizer = torch.optim.Adam([patch], lr=lr)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            # 랜덤 위치, 크기, 각도로 패치를 변형하여 이미지에 적용
            for i in range(images.size(0)):
                x, y = np.random.randint(0, images.size(2) - patch_size), np.random.randint(0, images.size(3) - patch_size)
                angle = np.random.uniform(-30, 30)
                scale = np.random.uniform(0.5, 1.5)
                transformed_patch = patch_transform(patch, angle, scale, device)
                images[i] = patch_to_image(images[i], transformed_patch, x, y)

            preds = model(images)
            loss = compute_loss(preds, patch_target)
            loss.backward()
            optimizer.step()

        # 에포크마다 검증
        model.eval()
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                for i in range(images.size(0)):
                    x, y = np.random.randint(0, images.size(2) - patch_size), np.random.randint(0, images.size(3) - patch_size)
                    angle = np.random.uniform(-30, 30)
                    scale = np.random.uniform(0.5, 1.5)
                    transformed_patch = patch_transform(patch, angle, scale, device)
                    images[i] = patch_to_image(images[i], transformed_patch, x, y)

                preds = model(images)
                val_loss = compute_loss(preds, patch_target)

        print(f"Epoch: {epoch + 1}/{epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

        elapsed_time = time.time() - start_time
        remaining_time = (epochs - (epoch + 1)) * (elapsed_time / (epoch + 1))

        days = remaining_time // (24 * 3600)
        remaining_time = remaining_time % (24 * 3600)
        hours = remaining_time // 3600
        remaining_time %= 3600
        minutes = remaining_time // 60
        remaining_time %= 60
        seconds = remaining_time

        elapsed_minutes = int(elapsed_time // 60)
        elapsed_seconds = int(elapsed_time % 60)

        print(f"Elapsed Time: {elapsed_minutes}m {elapsed_seconds}s")
        print(f"Remaining Time: {int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s")

        patch_save(patch, save_path, f"epoch_{epoch + 1}")


if __name__ == '__main__':

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS 사용")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("CUDA 사용")
    else:
        device = torch.device("cpu")
        print("CPU 사용")

    patch_image = "../patch_origin.png"
    patch_size = 32
    patch_shape = "circle"
    train_image_path = "../model/datasets/coco/images/train2017/"
    val_image_path = "../model/datasets/coco/images/val2017/"
    patch_target = 0
    lr = 0.001
    epochs = 100
    batch_size = 16
    num_workers = 3

    patch = patch_init(patch_image, patch_size, patch_shape)
    model = load_model()

    patch_optimize()
