import os
import torch
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, image_paths, target_class_id, resize_to=(416, 416)):
        self.image_paths = [path for path in image_paths if
                            int(os.path.basename(path).split('_')[0]) == target_class_id]
        self.resize_to = resize_to
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize_to),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv.imread(self.image_paths[idx], cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = self.transform(image)
        return image


def load_model():
    print("모델 로드")
    model = YOLO('../yolov8n.pt', verbose=False).to(device)
    print("모델 로드 완료")
    return model


def patch_init(patch_image, patch_size, patch_shape):
    print("패치 초기화")
    image = cv.imread(patch_image, cv.IMREAD_UNCHANGED)
    if image is None:
        print("이미지 로드 실패. 랜덤 이미지 생성")
        image = np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)
    else:
        image = cv.resize(image, (patch_size, patch_size))

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

    alpha_channel = np.where(shape == 255, 255, 0).astype(np.uint8)
    image[:, :, 3] = alpha_channel

    patch = torch.from_numpy(image).float().to(device).permute(2, 0, 1) / 255.0
    patch.requires_grad_(True)
    print("패치 초기화 완료")

    patch_save(patch, "patch/", "initial_patch")

    return patch


def patch_random_init():
    image = np.random.randint(0, 255, (patch_size, patch_size, 4), dtype=np.uint8)

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

    alpha_channel = np.where(shape == 255, 255, 0).astype(np.uint8)
    image[:, :, 3] = alpha_channel

    patch = torch.from_numpy(image).float().to(device).permute(2, 0, 1) / 255.0
    patch.requires_grad_(True)

    return patch


def patch_to_image(image, patch, x, y):
    patch_height, patch_width = patch.shape[1:]

    patch_np = patch.permute(1, 2, 0).cpu().detach().numpy()
    patch_np = (patch_np * 255).astype(np.uint8)
    alpha_s = patch_np[:, :, 3] / 255.0
    patch_np = patch_np[:, :, :3]

    image_np = image.permute(1, 2, 0).cpu().detach().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        image_np[y:y + patch_height, x:x + patch_width, c] = (
                alpha_s * patch_np[:, :, c] + alpha_l * image_np[y:y + patch_height, x:x + patch_width, c]
        )

    image_np = image_np.astype(np.float32) / 255.0
    image = torch.tensor(image_np, dtype=torch.float32, device=device).permute(2, 0, 1)

    return image


def patch_transform(patch, angle, scale, device='cpu'):
    patch_np = patch.permute(1, 2, 0).cpu().detach().numpy()
    patch_np = (patch_np * 255).astype(np.uint8)

    original_height, original_width = patch_np.shape[:2]
    resized_width = int(original_width * scale)
    resized_height = int(original_height * scale)
    resized_patch_np = cv.resize(patch_np, (resized_width, resized_height), interpolation=cv.INTER_AREA)

    new_center = (resized_width // 2, resized_height // 2)
    matrix = cv.getRotationMatrix2D(new_center, angle, 1.0)
    rotated_patch_np = cv.warpAffine(resized_patch_np, matrix, (resized_width, resized_height))

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


def patch_generate(image, patch=None):
    global patch_target
    if image.dim() == 3:
        image = image.unsqueeze(0)

    angle = np.random.randint(0, 360)
    scale = np.random.uniform(0.8, 1.5)

    if patch is None:
        temp_patch = patch_random_init()
    else:
        temp_patch = patch
    transformed_temp_patch = patch_transform(temp_patch, angle, scale, device.type)

    x = np.random.randint(0, image.size(2) - transformed_temp_patch.size(1))
    y = np.random.randint(0, image.size(3) - transformed_temp_patch.size(2))

    image = patch_to_image(image[0], transformed_temp_patch, x, y)

    results = model(image.unsqueeze(0), verbose=False)

    is_target_detected = False
    for result in results:
        for box in result.boxes:
            if box.cls == patch_target:
                is_target_detected = True
                break
        if is_target_detected:
            break

    if is_target_detected:
        return results, False
    else:
        return results, True


def detection_loss(output, target_class_id):
    loss = torch.tensor(0.0, device=device)
    for result in output:
        for box in result.boxes:
            if box.cls == target_class_id:
                loss += -torch.log(box.conf)
    loss /= len(output)
    return loss


def train(train_loader, epoch):
    global patch
    total_loss = 0
    total_success = 0
    total_fail = 0

    for batch_index, images in enumerate(train_loader):
        batch_loss = 0
        batch_success = 0
        batch_fail = 0

        for i in range(images.size(0)):
            image = images[i].to(device)

            image = image.unsqueeze(0)
            results = model(image, verbose=False)

            is_target_detected = False
            for result in results:
                for box in result.boxes:
                    if box.cls == patch_target:
                        is_target_detected = True
                        break
                if is_target_detected:
                    break

            if not is_target_detected:
                continue

            output_origin, valid = patch_generate(image[0], patch)
            if valid:
                batch_success += 1
                continue

            iter = 0

            before_patch = patch.clone().detach()

            while True:
                output, valid = patch_generate(image[0])
                if valid:
                    loss = detection_loss(output, patch_target)
                    loss.backward()
                    batch_loss += loss.item()
                    patch -= lr * patch.grad
                    patch.grad.zero_()
                    batch_success += 1
                    print("패치 생성 성공")
                    break
                if iter > max_iter:
                    loss = detection_loss(output, patch_target)
                    batch_loss += loss.item()
                    batch_fail += 1
                    print("패치 생성 실패")
                    break
                iter += 1

            after_patch = patch.clone().detach()
            print("패치 변화량: ", torch.norm(before_patch - after_patch, p=2))

        print(
            f"[Train] Epoch: {epoch + 1}, Batch: {batch_index + 1}/{len(train_loader)}, Loss: {batch_loss / images.size(0)}")
        print(f"[Train] Success: {batch_success}, Fail: {batch_fail}")
        total_loss += batch_loss
        total_success += batch_success
        total_fail += batch_fail

    return total_loss, total_success, total_fail


def val(val_loader, epoch):
    global patch
    total_loss = 0
    total_success = 0
    total_fail = 0

    with torch.no_grad():
        for batch_index, images in enumerate(val_loader):
            batch_success = 0
            batch_fail = 0

            for i in range(images.size(0)):
                image = images[i].to(device)

                image = image.unsqueeze(0)
                results = model(image, verbose=False)

                is_target_detected = False
                for result in results:
                    for box in result.boxes:
                        if box.cls == patch_target:
                            is_target_detected = True
                            break
                    if is_target_detected:
                        break

                if not is_target_detected:
                    continue

                output_origin, valid = patch_generate(image[0], patch)
                if valid:
                    batch_success += 1
                    print("패치 검증 성공")
                    continue
                batch_fail += 1
                print("패치 검증 실패")

            print(f"[Val] Epoch: {epoch + 1}, Batch: {batch_index + 1}/{len(val_loader)}")
            print(f"[Val] Success: {batch_success}, Fail: {batch_fail}")
            total_success += batch_success
            total_fail += batch_fail

    return total_loss, total_success, total_fail


def optimize_patch():
    global patch
    train_image_paths = [os.path.join(train_image_path, image_name) for image_name in os.listdir(train_image_path)]
    val_image_paths = [os.path.join(val_image_path, image_name) for image_name in os.listdir(val_image_path)]

    train_dataset = ImageDataset(train_image_paths, patch_target)
    val_dataset = ImageDataset(val_image_paths, patch_target)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    best_loss = float('inf')

    start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")

        train_loss, train_success, train_fail = train(train_loader, epoch)
        val_loss, val_success, val_fail = val(val_loader, epoch)

        print(
            f"[Train] Epoch: {epoch + 1}, Loss: {train_loss / len(train_loader)}, Success: {train_success}, Fail: {train_fail}")
        print(f"[Val] Epoch: {epoch + 1}, Loss: {val_loss / len(val_loader)}, Success: {val_success}, Fail: {val_fail}")

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patch_save(patch, "patch/", f"best_{epoch}_{val_loss}")

        # 패치 저장
        patch_save(patch, "patch/", f"last_{epoch}_{val_loss}")

        with open("patch/patch_log.txt", "a") as f:
            f.write(f"[{epoch}]")
            f.write(f"{train_loss / len(train_loader)}, {train_success}, {train_fail}\n")
            f.write(f"{val_loss / len(val_loader)}, {val_success}, {val_fail}\n")
            f.write("\n")

        epoch_end_time = time.time()

        elapsed_time = epoch_end_time - start_time
        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        expected_time = elapsed_time / (epoch + 1) * (epochs - epoch - 1)
        expected_time_str = time.strftime("%d %H:%M:%S", time.gmtime(expected_time))
        print(f"Elapsed time: {elapsed_time_str}, Expected time: {expected_time_str}")


if __name__ == '__main__':
    global patch

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS 사용")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.backends.cudnn.benchmark = True
        print("CUDA 사용")
    else:
        device = torch.device("cpu")
        print("CPU 사용")

    patch_image = "../patch_origin.png"
    patch_size = 64
    patch_shape = "circle"
    train_image_path = "../image/single_label_train2017/"
    val_image_path = "../image/single_label_val2017/"
    patch_target = 0
    lr = 0.001
    epochs = 3000
    batch_size = 128
    num_workers = 3
    max_iter = 1000

    model = load_model()
    patch = patch_init(patch_image, patch_size, patch_shape)

    optimize_patch()
