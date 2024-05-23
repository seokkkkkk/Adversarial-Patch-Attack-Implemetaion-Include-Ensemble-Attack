import numpy as np
import cv2 as cv
import torch
import os
from ultralytics import YOLO


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
    elif patch_shape == "rectangle":
        cv.rectangle(shape, (0, 0), (patch_size, patch_size), 255, -1)
    else:
        raise ValueError("Invalid patch_shape")
    image = cv.bitwise_and(image, image, mask=shape)

    # 패치를 텐서로 변환
    patch = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    patch = patch.to(device)
    patch.requires_grad_(True)
    return patch


def save_patch(patch, name, save_path):
    # 패치를 이미지로 변환하여 저장
    patch_np = (patch.permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
    os.makedirs(save_path, exist_ok=True)
    cv.imwrite(os.path.join(save_path, f"{name}.png"), patch_np)
    print(f"Patch saved at {os.path.join(save_path, f'{name}.png')}")


def transform_patch(patch, angle, scale, device):
    # 패치를 회전 및 크기 조정
    patch_np = (patch.permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
    resized_patch = cv.resize(patch_np, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    center = (resized_patch.shape[1] // 2, resized_patch.shape[0] // 2)
    matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    transformed_patch = cv.warpAffine(resized_patch, matrix, (resized_patch.shape[1], resized_patch.shape[0]))
    transformed_patch = torch.from_numpy(transformed_patch).permute(2, 0, 1).float() / 255.0
    transformed_patch = transformed_patch.to(device)
    transformed_patch.requires_grad_(True)
    return transformed_patch


def apply_patch_to_image(image, patch, x, y):
    # 이미지에 패치 적용
    patched_image = image.clone()
    patched_image[:, y:y + patch.shape[1], x:x + patch.shape[2]] = patch
    return patched_image.unsqueeze(0)


def random_transformation():
    # 패치의 회전 및 크기 조정을 위한 랜덤 값 생성
    angle = np.random.randint(-180, 180)
    scale = np.random.uniform(0.5, 1.5)
    return angle, scale


def split_dataset(images_path, max_images):
    # 데이터셋 분할
    images = [os.path.join(root, file) for root, _, files in os.walk(images_path) for file in files if
              file.endswith(".jpg")]
    np.random.shuffle(images)
    images = images[:max_images] if len(images) > max_images else images
    train_split = int(len(images) * 0.8)
    return images[:train_split], images[train_split:]


def preprocess_image(image_path, device):
    # 이미지 전처리
    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    image = cv.resize(image, (image.shape[1] // 32 * 32, image.shape[0] // 32 * 32))
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image = image.to(device)
    image.requires_grad_(True)
    return image


def training_log(epoch, epochs, train_loss, val_loss, path):
    # 학습 로그 저장
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("epoch,epochs,train_loss,val_loss\n")
    with open(path, "a") as f:
        f.write(f"{epoch},{epochs},{train_loss},{val_loss}\n")


def train(model, optimizer, initial_patch, images, target_class, device):
    # 패치 학습 과정
    train_loss = 0
    for image_path in images:
        image = preprocess_image(image_path, device)
        angle, scale = random_transformation()
        transformed_patch = transform_patch(initial_patch, angle, scale, device)
        x = np.random.randint(0, image.shape[2] - transformed_patch.shape[2])
        y = np.random.randint(0, image.shape[1] - transformed_patch.shape[1])
        patched_image = apply_patch_to_image(image, transformed_patch, x, y)
        result = model(patched_image)
        result = result[0].probs.data.unsqueeze(0)
        loss = -torch.nn.functional.log_softmax(result, dim=1)[:, target_class].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(images)
    return train_loss


def val(model, initial_patch, images, target_class, device):
    # 학습된 패치 검증
    val_loss = 0
    for image_path in images:
        image = preprocess_image(image_path, device)
        angle, scale = random_transformation()
        transformed_patch = transform_patch(initial_patch, angle, scale, device)
        x = np.random.randint(0, image.shape[2] - transformed_patch.shape[2])
        y = np.random.randint(0, image.shape[1] - transformed_patch.shape[1])
        patched_image = apply_patch_to_image(image, transformed_patch, x, y)

        result = model(patched_image)
        result = result[0].probs.data.unsqueeze(0)
        loss = -torch.nn.functional.log_softmax(result, dim=1)[:, target_class].mean()

        val_loss += loss.item()
    val_loss /= len(images)
    return val_loss


def train_model(model, optimizer, initial_patch, train_images, val_images, epochs, target_class, device, stop_threshold):
    best_val_loss = float("inf")
    best_val_epoch = 0
    early_stopping = stop_threshold
    best_patch = initial_patch.clone().detach()
    for epoch in range(epochs):
        train_loss = train(model, optimizer, initial_patch, train_images, target_class, device)
        # no grad
        with torch.no_grad():
            val_loss = val(model, initial_patch, val_images, target_class, device)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        training_log(epoch, epochs, train_loss, val_loss, "training_log.csv")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_patch = initial_patch.clone().detach()
            save_patch(best_patch, f"best_patch_{epoch}_{best_val_loss:.4f}", "patch/")
        else:
            # val_loss가 개선되지 않으면 initial_patch를 이전 최적 패치로 되돌림
            initial_patch.data = best_patch.clone().detach()

        if epoch - best_val_epoch > early_stopping:
            print(f"Early stopping at epoch {epoch + 1}")
            return best_patch

    return best_patch


def main():
    # device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 객체 분류 모델 설정
    model = YOLO("yolov8n-cls.pt").to(device)

    # 초기 패치 및 학습 관련 설정
    patch_size = 64
    patch_shape = "circle"
    patch_save_path = "patch/"
    initial_patch = patch_init(patch_size, patch_shape, device)
    optimizer = torch.optim.Adam([initial_patch], lr=0.01)
    epochs = 100
    images_path = "image/image/"
    max_images = 1000
    target_class = 0
    stop_threshold = 10
    save_patch(initial_patch, "initial_patch", patch_save_path)

    # 데이터셋 분할
    train_images, val_images = split_dataset(images_path, max_images)

    # 패치 생성
    best_patch = train_model(model, optimizer, initial_patch, train_images, val_images, epochs, target_class, device, stop_threshold)

    # 최종 패치 저장
    save_patch(best_patch, "final_patch", patch_save_path)

    # 최종 패치 imshow
    patch_np = (best_patch.permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
    cv.imshow("final_patch", patch_np)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
