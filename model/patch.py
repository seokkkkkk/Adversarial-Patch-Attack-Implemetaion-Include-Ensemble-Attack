import cv2 as cv
import numpy as np
import torch
import os

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
