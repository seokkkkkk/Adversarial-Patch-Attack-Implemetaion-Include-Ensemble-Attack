from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2 as cv
import torch
import time
import os


class ImageDataset(Dataset):
    def __init__(self, image_paths, device, img_size=(416, 416)):
        self.image_paths = image_paths
        self.device = device
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image at path: {image_path}")

        if image.ndim == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        image = cv.resize(image, self.img_size)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = image.to(self.device)
        return image

def patch_init(patch_size, patch_shape, device, custom_patch_path=None):
    # 사용자 지정 패치 경로가 있을 경우 이미지 로드, 없을 경우 랜덤 패치 생성
    if custom_patch_path:
        image = cv.imread(custom_patch_path, cv.IMREAD_UNCHANGED)
        image = cv.resize(image, (patch_size, patch_size))
    else:
        image = np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)

    # 패치 모양 설정
    shape = np.zeros((patch_size, patch_size), dtype=np.uint8)
    center = (patch_size // 2, patch_size // 2)
    if patch_shape == "circle":
        radius = patch_size // 2
        cv.circle(shape, center, radius, 255, -1)
    elif patch_shape == "default":
        cv.rectangle(shape, (0, 0), (patch_size, patch_size), 255, -1)
    else:
        raise ValueError("Invalid patch_shape")
    image = cv.bitwise_and(image, image, mask=shape)

    # 패치를 텐서로 변환 및 normalize
    patch = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    patch = patch.to(device).unsqueeze(0)
    patch.requires_grad_(True)
    return patch


def save_patch(patch, name, save_path):
    # 패치를 이미지로 변환하여 저장
    patch_np = (patch.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
    os.makedirs(save_path, exist_ok=True)
    cv.imwrite(os.path.join(save_path, f"{name}.png"), patch_np)
    print(f"Patch saved at {os.path.join(save_path, f'{name}.png')}")


def transform_patch(patch, angle, scale, device):
    # 패치를 회전 및 크기 조정
    patch = patch.clone()

    # 패치 이미지 numpy로 변환
    with torch.no_grad():
        patch_np = (patch.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    resized_patch = cv.resize(patch_np, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    center = (resized_patch.shape[1] // 2, resized_patch.shape[0] // 2)
    matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    transformed_patch_np = cv.warpAffine(resized_patch, matrix, (resized_patch.shape[1], resized_patch.shape[0]))

    # 패치 이미지 채널이 2차원인 경우 3차원으로 변경
    if transformed_patch_np.ndim == 2:
        transformed_patch_np = np.expand_dims(transformed_patch_np, axis=-1)
    if transformed_patch_np.shape[2] == 1:
        transformed_patch_np = np.repeat(transformed_patch_np, 3, axis=2)

    # 패치 이미지를 텐서로 변환 및 normalize
    transformed_patch = torch.from_numpy(transformed_patch_np).permute(2, 0, 1).float() / 255.0
    transformed_patch = transformed_patch.to(device)
    transformed_patch = transformed_patch.unsqueeze(0)

    return transformed_patch


def apply_patch_to_image(image, patch, x, y):
    # 이미지에 패치 적용
    patched_image = image.clone()
    patched_image[:, :, x:x + patch.shape[2], y:y + patch.shape[3]] = patch

    return patched_image


def random_transformation():
    # 패치의 회전 및 크기 조정을 위한 랜덤 값 생성
    angle = np.random.randint(-180, 180)
    scale = np.random.uniform(0.5, 1.5)
    return angle, scale


def split_dataset(images_path, max_images):
    # 데이터셋 분할
    # images 배열에 넣기 전에 이미지가 유효한지 판단
    images = [os.path.join(images_path, image) for image in os.listdir(images_path) if
              image.endswith((".jpg", ".JPEG", ".png"))]
    np.random.shuffle(images)
    images = images[:max_images] if len(images) > max_images else images
    train_split = int(len(images) * 0.8)
    return images[:train_split], images[train_split:]


def preprocess_image(image_path, device):
    # 이미지 전처리
    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image at path: {image_path}")

    # 이미지 채널이 1인 경우 3채널로 변경
    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    image = cv.resize(image, (image.shape[1] // 32 * 32, image.shape[0] // 32 * 32))
    # 이미지를 텐서로 변환 및 normalize
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image = image.to(device)
    image.requires_grad_(True)
    return image


def training_log(epoch, epochs, train_loss, val_loss, train_success, val_success, path):
    # 학습 로그 저장
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("epoch,epochs,train_loss,val_loss,train_success,val_success\n")
    with open(path, "a") as f:
        f.write(f"{epoch},{epochs},{train_loss},{val_loss},{train_success},{val_success}\n")


def plot_training_log(path):
    data = pd.read_csv(path)
    plt.figure(figsize=(10, 5))

    ax1 = plt.gca()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.plot(data["epoch"], data["train_loss"], label="Train Loss", color="tab:blue")
    ax1.plot(data["epoch"], data["val_loss"], label="Val Loss", color="tab:orange")
    ax1.legend(loc="upper left")
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.set_ylabel("Success Rate")
    ax2.plot(data["epoch"], data["train_success"], label="Train Success Rate", color="tab:green")
    ax2.plot(data["epoch"], data["val_success"], label="Val Success Rate", color="tab:red")
    ax2.legend(loc="upper right")

    plt.title("Training and Validation Loss and Success Rate Over Epochs")
    plt.show()
    plt.clf()  # 현재 플롯 지우기
    plt.close()  # 현재 플롯 닫기


def calculate_success(result, target_class):
    # 성공률 계산 (타겟 클래스가 예측된 경우)
    top1_class = torch.argmax(result, dim=1)
    success = (top1_class == target_class).float().mean().item()
    return success


def train(model, train_loader, target_class, device):
    # 학습
    train_loss = 0
    train_success = 0
    total_batches = len(train_loader)
    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)
        for image in images:
            image = image.unsqueeze(0)
            patch = initial_patch.clone()

            x = torch.randint(0, image.shape[2] - patch.shape[2], (1,), device=device).item()
            y = torch.randint(0, image.shape[3] - patch.shape[3], (1,), device=device).item()
            patched_image = apply_patch_to_image(image, patch, x, y)

            result = model(patched_image, verbose=False)
            result = result[0].probs.data.unsqueeze(0)

            target_prob = torch.nn.functional.log_softmax(result, dim=1)[:, target_class]
            loss = -target_prob.mean()
            success = calculate_success(result, target_class)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            initial_patch.data = torch.clamp(initial_patch, 0, 1)

            train_loss += loss.item()
            train_success += success

        # 각 배치마다 진행 상황 표시
        if (batch_idx + 1) % 100 == 0:
            batch_progress = (batch_idx + 1) * total_batches
            print(f"[Batch] {batch_progress}/{len(train_loader.dataset)}")

    train_loss /= len(train_loader.dataset)
    train_success /= len(train_loader.dataset)
    return train_loss, train_success


def val(model, val_loader, target_class, device):
    # 검증
    val_loss = 0
    val_success = 0
    total_batches = len(val_loader)
    for batch_idx, images in enumerate(val_loader):
        images = images.to(device)
        for image in images:
            image = image.unsqueeze(0)
            angle, scale = random_transformation()
            patch = initial_patch.clone().detach()
            transformed_patch = transform_patch(patch, angle, scale, device)
            x = torch.randint(0, image.shape[2] - transformed_patch.shape[2], (1,), device=device).item()
            y = torch.randint(0, image.shape[3] - transformed_patch.shape[3], (1,), device=device).item()
            patched_image = apply_patch_to_image(image, transformed_patch, x, y)

            result = model(patched_image, verbose=False)
            result = result[0].probs.data.unsqueeze(0)
            loss = -torch.nn.functional.log_softmax(result, dim=1)[:, target_class].mean()
            success = calculate_success(result, target_class)

            val_loss += loss.item()
            val_success += success

        # 각 배치마다 진행 상황 표시
        if (batch_idx + 1) % 100 == 0:
            batch_progress = (batch_idx + 1) * total_batches
            print(f"[Batch] {batch_progress}/{len(val_loader.dataset)}")

    val_loss /= len(val_loader.dataset)
    val_success /= len(val_loader.dataset)
    return val_loss, val_success


def train_patch(model, train_loader, val_loader, epochs, target_class, device, stop_threshold):
    # 패치 학습
    best_val_loss = float("inf")
    best_val_epoch = 0
    early_stopping = stop_threshold
    best_patch = initial_patch.clone().detach()
    start_time = time.time()

    for epoch in range(epochs):
        print(f"{'=' * 20} Epoch {epoch + 1}/{epochs} {'=' * 20}")
        train_loss, train_success = train(model, train_loader, target_class, device)

        with torch.no_grad():
            val_loss, val_success = val(model, val_loader, target_class, device)

        print(f"[Epoch] {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Train Success: {train_success:.4f} - Val Success: {val_success:.4f}")

        training_log(epoch, epochs, train_loss, val_loss, train_success, val_success, "../training_log.csv")
        plot_training_log("../training_log.csv")

        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (epoch + 1)) * (epochs - (epoch + 1))
        print(f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} - Remaining Time: {time.strftime('%H:%M:%S', time.gmtime(remaining_time))}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_patch = initial_patch.clone().detach()
            save_patch(best_patch, f"best_patch_{epoch}_{best_val_loss:.4f}", "../patch/")
        else:
            initial_patch.data = best_patch.clone().detach()

        if epoch - best_val_epoch > early_stopping:
            print(f"Early stopping at epoch {epoch + 1}")
            return best_patch

    return best_patch


def main():
    print("적대적 패치 생성 시작")

    global initial_patch
    global optimizer

    # device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # 객체 분류 모델 설정
    model = YOLO("../model/yolov8n-cls.pt").to(device)

    print("Model 로딩 완료")

    # 초기 패치 및 학습 관련 설정
    patch_size = 64
    patch_shape = "default"
    custom_patch_path = "../model/initial_custom_patch.png"
    patch_save_path = "../patch/"
    initial_patch = patch_init(patch_size, patch_shape, device, custom_patch_path)
    learning_rate = 0.001

    optimizer = torch.optim.Adam([initial_patch], lr=learning_rate)

    epochs = 100  # 학습 횟수 설정
    target_class = 107   # 타겟 클래스 (다른 이미지들을 타겟 클래스로 인식하도록 설정) jellyfish: 107
    stop_threshold = 10

    save_patch(initial_patch, "initial_patch", patch_save_path)

    # 데이터셋 분할
    batch_size = 64  # 배치 사이즈 설정
    max_images = batch_size * 1  # 학습할 최대 이미지 수
    images_path = "../datasets/imagenet/test"
    train_images, val_images = split_dataset(images_path, max_images)

    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")

    # 데이터로더 설정
    num_workers = 4
    train_loader = DataLoader(ImageDataset(train_images, device), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(ImageDataset(val_images, device), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("학습 시작")

    # 패치 생성
    best_patch = train_patch(model, train_loader, val_loader, epochs, target_class, device, stop_threshold)

    print("학습 완료")

    # 최종 패치 저장
    save_patch(best_patch, "final_patch", patch_save_path)

    # 최종 패치 imshow
    patch_np = (best_patch.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
    cv.imshow("final_patch", patch_np)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
