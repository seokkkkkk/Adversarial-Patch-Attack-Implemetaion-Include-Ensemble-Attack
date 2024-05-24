import torch
import time
from patch import apply_patch_to_image, random_transformation, transform_patch, save_patch
from utils import training_log, plot_training_log


def calculate_success(result, target_class):
    # 성공률 계산 (타겟 클래스가 예측된 경우)
    top1_class = torch.argmax(result, dim=1)
    success = (top1_class == target_class).float().mean().item()
    return success


def train(model, train_loader, target_class, device, initial_patch, optimizer):
    print("[Train]")
    # 학습
    train_loss = 0
    train_success = 0
    total_batches = len(train_loader)
    batch_progress = 0
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

            batch_progress += 1

        # 각 배치마다 진행 상황 표시
        if (batch_idx + 1) % 10 == 0:
            print(f"[Batch] {batch_progress}/{len(train_loader.dataset)}")

    train_loss /= len(train_loader.dataset)
    train_success /= len(train_loader.dataset)
    return train_loss, train_success


def val(model, val_loader, target_class, device, initial_patch):
    print("[Validation]")
    # 검증
    val_loss = 0
    val_success = 0
    batch_progress = 0
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

            batch_progress += 1

        # 각 배치마다 진행 상황 표시
        if (batch_idx + 1) % 10 == 0:
            print(f"[Batch] {batch_progress}/{len(val_loader.dataset)}")

    val_loss /= len(val_loader.dataset)
    val_success /= len(val_loader.dataset)
    return val_loss, val_success


def train_patch(model, train_loader, val_loader, epochs, target_class, device, stop_threshold, initial_patch,
                optimizer):
    # 패치 학습
    best_val_loss = float("inf")
    best_val_epoch = 0
    early_stopping = stop_threshold
    best_patch = initial_patch.clone().detach()
    start_time = time.time()

    for epoch in range(epochs):
        print(f"{'=' * 20} Epoch {epoch + 1}/{epochs} {'=' * 20}")
        train_loss, train_success = train(model, train_loader, target_class, device, initial_patch, optimizer)

        with torch.no_grad():
            val_loss, val_success = val(model, val_loader, target_class, device, initial_patch)

        print(
            f"[Epoch] {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Train Success: {train_success:.4f} - Val Success: {val_success:.4f}")

        training_log(epoch, epochs, train_loss, val_loss, train_success, val_success, "data/training_log.csv")
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
        else:
            initial_patch.data = best_patch.clone().detach()

        if epoch - best_val_epoch > early_stopping:
            print(f"Early stopping at epoch {epoch + 1}")
            return best_patch

    return best_patch
