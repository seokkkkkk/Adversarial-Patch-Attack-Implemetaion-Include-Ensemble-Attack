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