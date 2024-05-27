import torch
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from torch.utils.data import DataLoader
from dataset import ImageDataset
from patch import patch_init, save_patch
from utils import split_dataset, return_path_to_images
from train import train_patch

def main():
    print("적대적 패치 생성 시작")

    global initial_patch
    global optimizer

    # device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # 객체 분류 모델 설정
    model = YOLO("yolov8s-cls.pt").to(device)

    print("Model 로딩 완료")

    # 초기 패치 및 학습 관련 설정
    patch_size = 50
    patch_shape = "default"
    custom_patch_path = None # 필요한 경우 "path/to/custom/patch.png"
    patch_save_path = "patch/"
    initial_patch = patch_init(patch_size, patch_shape, device, custom_patch_path)
    learning_rate = 0.001

    optimizer = torch.optim.Adam([initial_patch], lr=learning_rate)

    epochs = 1000  # 학습 횟수 설정
    target_class = 859 # 공격 target class 설정
    stop_threshold = 20

    save_patch(initial_patch, "initial_patch", patch_save_path)

    # 데이터셋 분할
    batch_size = 300  # 배치 사이즈 설정을 1로 변경
    max_images = 40000  # 학습할 최대 이미지 수
    images_path = "images/path/to/attack"  # 공격할 대상 이미지 경로
    train_images, val_images = split_dataset(return_path_to_images(images_path), max_images)

    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")

    # 데이터로더 설정
    train_loader = DataLoader(ImageDataset(train_images, device), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ImageDataset(val_images, device), batch_size=batch_size, shuffle=False, num_workers=0)

    print("학습 시작")

    # 패치 생성
    best_patch = train_patch(model, train_loader, val_loader, epochs, target_class, device, stop_threshold, initial_patch, optimizer)

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
