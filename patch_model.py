import os
import torch
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt

# device 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS를 사용합니다.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA를 사용합니다.")
else:
    device = torch.device("cpu")
    print("CPU를 사용합니다.")


class ImageDataset(Dataset):
    def __init__(self, image_paths, resize_to=(416, 416)):
        self.image_paths = image_paths
        self.resize_to = resize_to

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv.imread(self.image_paths[idx], cv.IMREAD_COLOR)  # BGR로 읽음
        image = cv.resize(image, self.resize_to, interpolation=cv.INTER_AREA)  # 이미지를 resize_to 크기로 리사이즈
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # [H, W, C] -> [C, H, W]
        return image


def load_model():
    print("모델 로드")
    model = YOLO('yolov8n.pt', verbose=False).to(device)
    print("모델 로드 완료")
    return model


def load_images_from_directory(directory_path):
    print(f"{directory_path} 에서 이미지 로드")
    images = []
    for filename in os.listdir(directory_path):
        image_path = os.path.join(directory_path, filename)
        image_temp = cv.imread(image_path)
        if image_temp is None:
            print(f"{image_path} 이미지 로드 실패")
            continue
        image_temp = cv.cvtColor(image_temp, cv.COLOR_BGR2RGB)
        # 이미지 크기를 stride에 맞춰 조정
        h, w, _ = image_temp.shape
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32
        image_temp = cv.resize(image_temp, (new_w, new_h))
        image_temp = torch.tensor(image_temp, dtype=torch.float32).permute(2, 0, 1) / 255.0
        images.append(image_temp)
    print("이미지 로드 완료")
    return images


def init_optimizer(patch, lr):
    print("옵티마이저 초기화")
    optimizer = torch.optim.Adam([patch], lr=lr)
    print("옵티마이저 초기화 완료")
    return optimizer


def patch_init(patch_image, patch_size, patch_shape):
    print("패치 초기화")
    # 이미지를 로드하고 패치 사이즈로 줄이고 모양을 변경
    image = cv.imread(patch_image, cv.IMREAD_UNCHANGED)
    if image is None:
        # 랜덤으로 이미지 생성
        print("이미지 로드 실패. 랜덤 이미지 생성")
        image = np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)
    else:
        image = cv.resize(image, (patch_size, patch_size))

    # 알파 채널 추가
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

    # 알파 채널을 사용하여 검은색 배경을 투명하게 설정
    alpha_channel = np.where(shape == 255, 255, 0).astype(np.uint8)
    image[:, :, 3] = alpha_channel

    patch = torch.from_numpy(image).float().to(device).permute(2, 0, 1) / 255.0
    patch.requires_grad_(True)
    print("패치 초기화 완료")

    patch_save(patch, "patch/", "initial_patch")

    return patch


def patch_to_image(image, patch, x, y):
    patch_height, patch_width = patch.shape[1:]

    # 패치 이미지를 numpy로 변환
    patch_np = patch.permute(1, 2, 0).cpu().detach().numpy()
    patch_np = (patch_np * 255).astype(np.uint8)

    # 패치의 알파 채널
    alpha_s = patch_np[:, :, 3] / 255.0
    patch_np = patch_np[:, :, :3]

    # 이미지도 numpy로 변환
    image_np = image.permute(1, 2, 0).cpu().detach().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    # 패치 배경 투명화 처리
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        image_np[y:y + patch_height, x:x + patch_width, c] = (
                alpha_s * patch_np[:, :, c] + alpha_l * image_np[y:y + patch_height, x:x + patch_width, c]
        )

    # 텐서로 변환
    image_np = image_np.astype(np.float32) / 255.0
    image = torch.tensor(image_np, dtype=torch.float32, device=device).permute(2, 0, 1)

    return image


def patch_transform(patch, angle, scale, device='cpu'):
    patch_np = patch.permute(1, 2, 0).cpu().detach().numpy()
    patch_np = (patch_np * 255).astype(np.uint8)

    # 원래 크기 가져오기
    original_height, original_width = patch_np.shape[:2]

    # 패치 크기 조절
    resized_width = int(original_width * scale)
    resized_height = int(original_height * scale)
    resized_patch_np = cv.resize(patch_np, (resized_width, resized_height), interpolation=cv.INTER_AREA)

    # 크기 조절된 패치의 새로운 중심 계산
    new_center = (resized_width // 2, resized_height // 2)

    # 패치 회전 (원래 크기로 변환하지 않음)
    matrix = cv.getRotationMatrix2D(new_center, angle, 1.0)
    rotated_patch_np = cv.warpAffine(resized_patch_np, matrix, (resized_width, resized_height))

    # 다시 텐서로 변환
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


def loss_function_for_yolo(patch_target, results):
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    for result in results:
        boxes = result.boxes  # 탐지 결과의 박스 정보
        for box in boxes:
            if box.cls == patch_target:  # 탐지된 클래스가 목표 클래스와 일치하는지 확인
                loss = loss - box.conf  # 신뢰도에 기반한 손실 계산 (신뢰도가 높을수록 손실이 작아짐)
            else:
                loss = loss + box.conf  # 신뢰도가 높을수록 손실이 커짐
    return loss


def process_images(images, patch, model, patch_target, optimizer=None):
    total_loss = 0.0
    for i in range(images.size(0)):
        image = images[i]
        image_height, image_width = image.shape[1:]

        # 패치 변환 (회전, 크기 조절)
        angle = np.random.randint(0, 360)
        scale = np.random.uniform(0.8, 1.2)

        patch_transformed = patch_transform(patch, angle, scale)

        # 패치 위치 설정 (랜덤, patch_size와 image 사이즈를 고려하여 이미지에 넘치지 않도록)
        patch_height, patch_width = patch_transformed.shape[1:]

        x = np.random.randint(0, image_width - patch_width)
        y = np.random.randint(0, image_height - patch_height)

        # 패치 적용
        images[i] = patch_to_image(image, patch_transformed, x, y)

    # 이미지를 모델에 입력
    results = model(images, verbose=False)

    # 손실 함수 계산
    loss = loss_function_for_yolo(patch_target, results)
    total_loss += loss.item()

    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss


def patch_optimize(patch_image, patch_size, patch_shape, train_image_path, val_image_path, patch_target, lr, epochs,
                   batch_size, save_path="patch/"):
    
    print("패치 최적화 시작")

    patch = patch_init(patch_image, patch_size, patch_shape)

    model = load_model()

    optimizer = init_optimizer(patch, lr)

    train_image_paths = [os.path.join(train_image_path, f) for f in os.listdir(train_image_path)]
    val_image_paths = [os.path.join(val_image_path, f) for f in os.listdir(val_image_path)]

    train_dataset = ImageDataset(train_image_paths)
    val_dataset = ImageDataset(val_image_paths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_loss = float('inf')
    train_losses = []
    val_losses = []
    start_time = time.time()

    for epoch in range(epochs):
        # 패치 최적화
        total_loss = 0.0
        index = 0
        for images in train_loader:
            images = images.to(device)
            total_loss += process_images(images, patch, model, patch_target, optimizer)
            # 10회 반복마다 출력
            if index % 100 == 0:
                print(f"Batch: {index + 1}/{len(train_loader)}")
            index += 1
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch + 1}/{epochs}, Training Loss: {avg_loss}")

        # 패치 검증
        val_loss = 0.0
        with torch.no_grad():
            index = 0
            for images in val_loader:
                images = images.to(device)
                val_loss += process_images(images, patch, model, patch_target)
                # 100회 반복마다 출력
                if index % 100 == 0:
                    print(f"Batch: {index + 1}/{len(train_loader)}")
                index += 1
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch: {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")

        # 가장 낮은 손실을 기록하고 해당 패치를 저장
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_patch = patch.clone().detach()
            os.makedirs(save_path, exist_ok=True)
            patch_save(best_patch, save_path, f"best_patch_epoch_{epoch}")

        # 남은 시간 예측 및 출력
        elapsed_time = time.time() - start_time
        remaining_time = (epochs - (epoch + 1)) * (elapsed_time / (epoch + 1))
        print(f"현재 진행 시간: {elapsed_time} 분 / 남은 예상 시간: {remaining_time / 60:.2f} 분")

    # 손실 시각화
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.show()


# 예시 실행
patch_images = "img.png"
patch_size = 32
patch_shape = "circle"
train_image_path = "image/train/"
val_image_path = "image/val/"
patch_target = 0
lr = 0.001
epochs = 1000
batch_size = 1

patch_optimize(patch_images, patch_size, patch_shape, train_image_path, val_image_path, patch_target, lr, epochs, batch_size)
