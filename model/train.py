import torch
import time
from patch import apply_patch_to_image, random_transformation, transform_patch, save_patch
from utils import training_log, plot_training_log
import numpy as np
import cv2 as cv
from torchvision import transforms


def calculate_success(result, goal_class):
    # 성공률 계산 (타겟 클래스가 예측된 경우)
    top1_class = torch.argmax(result, dim=1)
    success = (top1_class == goal_class).float().mean().item()
    return success


# 변환 파이프라인 정의
transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),  # NumPy 배열을 PIL 이미지로 변환
    transforms.RandomRotation(degrees=45),  # 랜덤 회전 (각도 범위 조정 가능)
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # 랜덤 확대/축소 및 크롭
    transforms.ToTensor(),  # PIL 이미지를 텐서로 변환
])


def preprocess_image(image_np):
    # NumPy 배열을 변환 파이프라인을 사용하여 변환
    image_tensor = transform_pipeline(image_np)
    return image_tensor


def train_step(model, images, goal_class, target_class, device, patch, optimizer):
    train_loss = 0
    train_success = 0
    batch_size = images.size(0)

    for image in images:
        image_np = (image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
        image = preprocess_image(image_np).unsqueeze(0).to(device)
        current_patch = patch.clone()

        # 랜덤 변환
        angle, scale = random_transformation()
        transformed_patch = transform_patch(current_patch, angle, scale, device, "default")

        # 랜덤 위치에 패치 적용
        x = torch.randint(0, image.shape[2] - transformed_patch.shape[2], (1,), device=device).item()
        y = torch.randint(0, image.shape[3] - transformed_patch.shape[3], (1,), device=device).item()
        patched_image = apply_patch_to_image(image, transformed_patch, x, y)

        # 모델 예측
        result = model(patched_image, verbose=True)

        log_probs = torch.log(result[0].probs.data)
        goal_prob = log_probs.unsqueeze(0)
        loss = torch.nn.functional.nll_loss(goal_prob, torch.tensor([goal_class], device=device))
        success = calculate_success(goal_prob, goal_class)

        if result[0].probs.top1 == target_class:

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 패치 값을 0과 1 사이로 클리핑
            with torch.no_grad():
                patch.data = torch.clamp(patch, 0, 1)

        elif result[0].probs.top1 == goal_class:
            # 이미지 imshow
            patched_image_np = (patched_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
            patched_image_np = cv.cvtColor(patched_image_np, cv.COLOR_RGB2BGR)
            cv.imshow("patched_image", patched_image_np)
            cv.waitKey(1)

        train_loss += loss.item()
        train_success += success

    train_loss /= batch_size
    train_success /= batch_size

    return train_loss, train_success


def train(model, train_loader, goal_class, target_class, device, initial_patch, optimizer):
    print("[Train]")
    train_loss = 0
    train_success = 0

    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)
        batch_loss, batch_success = train_step(model, images, goal_class, target_class, device, initial_patch, optimizer)
        train_loss += batch_loss
        train_success += batch_success

        # 각 배치마다 진행 상황 표시
        if (batch_idx + 1) % 10 == 0:
            print(
                f"[Batch] {batch_idx + 1}/{len(train_loader)} - Loss: {batch_loss:.4f} - Success: {batch_success:.4f}")

    train_loss /= len(train_loader)
    train_success /= len(train_loader)
    return train_loss, train_success


def val(model, val_loader, goal_class, target_class, device, initial_patch):
    print("[Validation]")
    val_loss = 0
    val_success = 0

    for batch_idx, images in enumerate(val_loader):
        images = images.to(device)
        batch_loss, batch_success = train_step(model, images, goal_class, target_class, device, initial_patch,
                                               None)  # optimizer 없이 수행
        val_loss += batch_loss
        val_success += batch_success

        # 각 배치마다 진행 상황 표시
        if (batch_idx + 1) % 10 == 0:
            print(f"[Batch] {batch_idx + 1}/{len(val_loader)} - Loss: {batch_loss:.4f} - Success: {batch_success:.4f}")

    val_loss /= len(val_loader)
    val_success /= len(val_loader)
    return val_loss, val_success


def train_patch(model, train_loader, val_loader, epochs, goal_class, target_class, device, stop_threshold, initial_patch,
                optimizer):
    # 패치 학습
    best_val_loss = float("inf")
    best_val_epoch = 0
    early_stopping = stop_threshold
    best_patch = initial_patch.clone().detach()
    start_time = time.time()

    for epoch in range(epochs):
        print(f"{'=' * 20} Epoch {epoch + 1}/{epochs} {'=' * 20}")
        train_loss, train_success = train(model, train_loader, goal_class, target_class, device, initial_patch, optimizer)

        with torch.no_grad():
            val_loss, val_success = val(model, val_loader, goal_class, target_class, device, initial_patch)

        print(
            f"[Epoch] {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Train Success: {train_success:.4f} - Val Success: {val_success:.4f}")

        training_log(epoch, epochs, train_loss, val_loss, train_success, val_success,
                     "data/training_log.csv")
        plot_training_log("data/training_log.csv")

        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (epoch + 1)) * (epochs - (epoch + 1))
        print(
            f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} - Remaining Time: {time.strftime('%H:%M:%S', time.gmtime(remaining_time))}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_patch = initial_patch.clone().detach()
            save_patch(best_patch, f"best_patch_{epoch}_{best_val_loss:.4f}", "patch/")

        if epoch - best_val_epoch > early_stopping:
            print(f"Early stopping at epoch {epoch + 1}")
            return best_patch

    return best_patch
