import os
import pandas as pd
import numpy as np
import cv2 as cv
import torch
from matplotlib import pyplot as plt

def split_dataset(images, max_images):
    # 데이터셋 분할
    print("Total images found:", len(images))  # 전체 이미지 수 출력
    if len(images) == 0:
        raise ValueError("No images found in the specified directories.")

    print("Before shuffle:", images[:10])  # 처음 10개만 출력
    np.random.shuffle(images)
    print("After shuffle:", images[:10])  # 처음 10개만 출력
    images = images[:max_images] if len(images) > max_images else images
    train_split = int(len(images) * 0.8)
    return images[:train_split], images[train_split:]

def return_path_to_images(images_path):
    # 이미지 경로 반환
    images = []
    for root, dirs, files in os.walk(images_path):
        print(f"Current directory: {root}")  # 현재 디렉토리 출력
        for file in files:
            if file.endswith((".jpg", ".JPEG", ".png")):
                images.append(os.path.join(root, file))
                print(f"Found image: {os.path.join(root, file)}")  # 찾은 이미지 경로 출력
    return images


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

def plot_training_log(batch_log_filename, epoch_log_filename):
    batch_log_data = pd.read_csv(batch_log_filename)
    epoch_log_data = pd.read_csv(epoch_log_filename)

    epochs = epoch_log_data['epoch'].unique()
    train_losses_epoch = epoch_log_data['train_loss'].values
    val_losses_epoch = epoch_log_data['val_loss'].values
    train_successes_epoch = epoch_log_data['train_success'].values
    val_successes_epoch = epoch_log_data['val_success'].values

    plt.figure(figsize=(14, 10))

    # Plotting Batch-Level Loss
    plt.subplot(2, 2, 1)
    plt.plot(batch_log_data['batch_idx'], batch_log_data['train_loss'], label='Train Loss (Batch)', marker='o')
    plt.plot(batch_log_data['batch_idx'], batch_log_data['val_loss'], label='Validation Loss (Batch)', marker='o')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Batch)')
    plt.legend()

    # Plotting Batch-Level Success Rate
    plt.subplot(2, 2, 2)
    plt.plot(batch_log_data['batch_idx'], batch_log_data['train_success'], label='Train Success (Batch)', marker='o')
    plt.plot(batch_log_data['batch_idx'], batch_log_data['val_success'], label='Validation Success (Batch)', marker='o')
    plt.xlabel('Batch Index')
    plt.ylabel('Success Rate')
    plt.title('Training and Validation Success Rate (Batch)')
    plt.legend()

    # Plotting Epoch-Level Loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_losses_epoch, label='Train Loss (Epoch)', marker='o')
    plt.plot(epochs, val_losses_epoch, label='Validation Loss (Epoch)', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Epoch)')
    plt.legend()

    # Plotting Epoch-Level Success Rate
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_successes_epoch, label='Train Success (Epoch)', marker='o')
    plt.plot(epochs, val_successes_epoch, label='Validation Success (Epoch)', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    plt.title('Training and Validation Success Rate (Epoch)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def batch_training_log(epoch, batch_idx, num_batches, train_loss, val_loss, train_success, val_success, filename):
    log_data = pd.DataFrame({
        'epoch': [epoch],
        'batch_idx': [batch_idx],
        'num_batches': [num_batches],
        'train_loss': [train_loss],
        'val_loss': [val_loss],
        'train_success': [train_success],
        'val_success': [val_success]
    })

    if not os.path.isfile(filename):
        log_data.to_csv(filename, index=False)
    else:
        log_data.to_csv(filename, mode='a', header=False, index=False)

def epoch_training_log(epoch, train_loss, val_loss, train_success, val_success, filename):
    log_data = pd.DataFrame({
        'epoch': [epoch],
        'train_loss': [train_loss],
        'val_loss': [val_loss],
        'train_success': [train_success],
        'val_success': [val_success]
    })

    if not os.path.isfile(filename):
        log_data.to_csv(filename, index=False)
    else:
        log_data.to_csv(filename, mode='a', header=False, index=False)