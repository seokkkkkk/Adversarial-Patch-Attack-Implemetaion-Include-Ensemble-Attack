import torch.nn.functional as F
import numpy as np
import cv2 as cv
import torch
import os

def patch_init(patch_size, patch_shape, device, custom_patch_path=None):
    # 사용자 지정 패치 경로가 있을 경우 이미지 로드, 없을 경우 랜덤 패치 생성
    if custom_patch_path:
        image = cv.imread(custom_patch_path, cv.IMREAD_COLOR)
        image = cv.resize(image, (patch_size, patch_size))
        # BGR을 RGB로 변환
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
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
    # rgb -> bgr
    patch_np = cv.cvtColor(patch_np, cv.COLOR_RGB2BGR)
    os.makedirs(save_path, exist_ok=True)
    cv.imwrite(os.path.join(save_path, f"{name}.png"), patch_np)
    print(f"Patch saved at {os.path.join(save_path, f'{name}.png')}")

def transform_patch(patch, angle, scale, device, patch_shape):
    # 패치를 회전 및 크기 조정
    patch = patch.clone()

    # 각도를 라디안으로 변환 및 Tensor로 변환
    angle = -angle * torch.pi / 180  # 회전 각도를 시계 반대 방향으로 설정
    angle = torch.tensor(angle, device=device)  # Tensor로 변환

    # 패치의 원래 크기를 가져옴
    original_size = patch.shape[2:]
    new_size = [int(dim * scale) for dim in original_size]

    # 크기 조정
    resized_patch = F.interpolate(patch, size=new_size, mode='bilinear', align_corners=True)

    # 회전
    theta = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0]], device=device, dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), resized_patch.size(), align_corners=True).float()
    transformed_patch = F.grid_sample(resized_patch, grid, align_corners=True, mode='bilinear')

    return transformed_patch

def apply_patch_to_image(image, patch, x, y):
    # 이미지에 패치 적용
    patched_image = image.clone()

    # 패치의 크기만큼 이미지에 덮어씌우기
    for c in range(3):
        patched_image[:, c, x:x + patch.shape[2], y:y + patch.shape[3]] = patch[:, c, :, :]

    return patched_image

def random_transformation():
    # 패치의 회전 및 크기 조정을 위한 랜덤 값 생성
    angle = np.random.choice([0, 90, 180, 270])
    scale = np.random.uniform(0.7, 1.2)
    return angle, scale