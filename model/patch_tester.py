import patch
import utils
import ultralytics
import dataset
import numpy as np
import cv2 as cv
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLOv8n-cls 모델 로드
model = ultralytics.YOLO("yolov8s-cls.pt").to(device)

# 이미지 폴더 경로
image_path = "C:/Users/HOME/Desktop/imagenet/ILSVRC/Data/CLS-LOC/test/"

image_path = utils.return_path_to_images(image_path)

# 이미지 폴더 내의 모든 이미지 파일 처리
test_loader = DataLoader(dataset.ImageDataset(image_path, device), batch_size=1, shuffle=False, num_workers=0)

test_patch = patch.patch_init(80, "default", device, "C:\\Users\\HOME\\IdeaProjects\\adversarial_patch\\model\\patch\\1\\49.png")

total_length = 0

results_correct = 0

transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),  # NumPy 배열을 PIL 이미지로 변환
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # 랜덤 확대/축소 및 크롭
    transforms.ToTensor(),  # PIL 이미지를 텐서로 변환
])


def preprocess_image(image_np):
    # NumPy 배열을 변환 파이프라인을 사용하여 변환
    image_tensor = transform_pipeline(image_np)
    return image_tensor

for images in test_loader:
    for image in images:
        image_np = (image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
        image = preprocess_image(image_np).unsqueeze(0).to(device)

        angle, scale = patch.random_transformation()
        test_patch_transformed = patch.transform_patch(test_patch, angle, scale, device, "default")
        x = torch.randint(0, image.shape[2] - test_patch_transformed.shape[2], (1,), device=device).item()
        y = torch.randint(0, image.shape[3] - test_patch_transformed.shape[3], (1,), device=device).item()
        patched_image = patch.apply_patch_to_image(image, test_patch_transformed, x, y)

        # imshow
        np_patched_image = patched_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0
        np_patched_image = np_patched_image.astype(np.uint8)
        np_patched_image = cv.cvtColor(np_patched_image, cv.COLOR_RGB2BGR)
        cv.imshow("patched_image", np_patched_image)
        cv.waitKey(1)

        results = model(patched_image, verbose=False)
        top1_class = results[0].probs.top1
        total_length += 1
        if top1_class == 950:
            results_correct += 1
        print(f"Current correct rate: {results_correct / total_length * 100:.2f}% Current Images: {total_length}")
