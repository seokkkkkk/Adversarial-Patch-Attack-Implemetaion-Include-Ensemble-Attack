from ultralytics import YOLO
import numpy as np
import cv2 as cv
import torch
import os

from torchvision import models


def patch_init(custom_patch_path=None):
    if custom_patch_path is not None:
        image = cv.imread(custom_patch_path, cv.IMREAD_UNCHANGED)
        image = cv.resize(image, (patch_size, patch_size))
    else:
        image = np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)


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

    image = cv.bitwise_and(image, image, mask=shape)

    patch = torch.from_numpy(image).float().to(device).permute(2, 0, 1) / 255.0
    patch.requires_grad_(True)
    return patch


def patch_save(name):
    print("최소 손실 패치 저장: " + name)
    patch_np = initial_patch.permute(1, 2, 0).cpu().detach().numpy() * 255.0
    patch_np = patch_np.astype(np.uint8)

    os.makedirs(patch_save_path, exist_ok=True)
    cv.imwrite(os.path.join(patch_save_path, f"{name}.png"), patch_np)
    print(f"{os.path.join(patch_save_path, f'{name}.png')}에 저장 완료")


def patch_transform(angle, scale, device="cuda" if torch.cuda.is_available() else "cpu"):
    patch_np = initial_patch.permute(1, 2, 0).cpu().detach().numpy()
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


def patch_to_image(image, patch, x, y):
    image = image.clone().detach()
    patch = patch.clone().detach()

    patch_width, patch_height = patch.shape[2], patch.shape[1]

    image[:, y:y + patch_height, x:x + patch_width] = patch

    return image.unsqueeze(0)


def transformation():
    angle = np.random.randint(-180, 180)
    scale = np.random.uniform(0.5, 1.5)
    return angle, scale


def train_val_split():
    images = []
    for root, dirs, files in os.walk(images_path):
        for file in files:
            if file.endswith(".jpg"):
                images.append(os.path.join(root, file))
    np.random.shuffle(images)

    if len(images) > max_images:
        images = images[:max_images]

    train = images[:int(len(images) * 0.8)]
    val = images[int(len(images) * 0.8):]

    return train, val


def image_process(image):
    image = cv.imread(image, cv.IMREAD_UNCHANGED)

    # 이미지의 크기를 32로 나누어 떨어지게 조정(직사각형 아니어도 됨)
    image_width = image.shape[1] // 32 * 32
    image_height = image.shape[0] // 32 * 32
    image = cv.resize(image, (image_width, image_height))

    image = torch.tensor(image, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
    image.requires_grad_(True)
    return image


def train(images):
    for image in images:
        image = image_process(image)

        angle, scale = transformation()
        patch_transformed = patch_transform(angle, scale)

        image_width, image_height = image.shape[2], image.shape[1]
        patch_width, patch_height = patch_transformed.shape[2], patch_transformed.shape[1]

        x = np.random.randint(0, image_width - patch_width)
        y = np.random.randint(0, image_height - patch_height)

        image = patch_to_image(image, patch_transformed, x, y)

        # model vgg16

        new_model = models.vgg16(pretrained=True).to(device)

        new_output = new_model(image)

        print(new_output)


        result = predict(image)
        result = result[0].probs.data.to(device)
        print(result)
        result = result.clone().detach().requires_grad_(True)

        target = torch.tensor([target_class], device=device)

        # loss = torch.nn.CrossEntropyLoss()(result, target)

        # vgg15 loss

        loss = torch.nn.CrossEntropyLoss()(new_output, target)

        # before backward

        before = initial_patch.clone().detach()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # after backward

        after = initial_patch.clone().detach()

        # diff

        print(torch.sum(torch.abs(after - before)))


def predict(image):
    result = model.predict(image, verbose=False)
    return result


def make_adv_patch():
    train_images, val_images = train_val_split()
    for epoch in range(epochs):
        train(train_images)


if __name__ == "__main__":
    global model

    global patch_size
    global patch_shape
    global patch_save_path
    global initial_patch

    global optimizer

    global epochs
    global batch_size
    global learning_rate

    global images_path
    global max_images

    global target_class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("yolov8n-cls.pt").to(device)

    patch_size = 16
    patch_shape = "circle"
    patch_save_path = "patch/"
    initial_patch = patch_init()
    initial_patch.requires_grad_(True)

    optimizer = torch.optim.Adam([initial_patch], lr=0.01)

    epochs = 100
    batch_size = 1
    learning_rate = 0.01

    images_path = "image/single_label_train2017/"
    max_images = 1000

    target_class = 0

    patch_save("initial_patch")
    make_adv_patch()


