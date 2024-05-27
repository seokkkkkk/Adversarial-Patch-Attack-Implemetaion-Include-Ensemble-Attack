
# Adversarial Patch
[Korean](./README.md)
## Introduction

Adversarial Patch is an attack technique on object recognition models, first introduced in the 2017 paper “Adversarial Patch” by Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, and Justin Gilmer. This technique involves inserting a patch into a specific part of an image, causing the object recognition model to either fail to recognize the object or misclassify it as a different object.

Previous adversarial examples used the Perturbation technique, which attacks images by inserting fine noise. This technique had a limitation in that it had to generate adversarial examples for each image, and it was difficult to apply them to real-world objects. However, adversarial patches differ in that they can perform attacks on multiple images or real-world objects with a single patch.

## Formula Explanation

![Formula](https://github.com/seokkkkkk/yolov8n-cls_adversarial_patch/assets/66684504/754a20ac-9905-4b2d-94d8-fb0293deb6d5)

- The goal of the formula is to maximize the probability that the classifier \( Pr \) recognizes the image with the patch \( p \), applied at a specific location \( l \) and with a certain transformation \( t \), as the target class \( y^ \).
- This is achieved by optimizing the average expectation over various images \( x \), transformations \( t \), and locations \( l \).

## Training Process

- The patch \( p \) is iteratively trained based on various images \( x \), transformations \( t \), and locations \( l \).
    - During the training process, the patch \( p \) is updated via gradient descent to maximize the probability of being classified as the target class \( y^ \).

## Principle

- Image classification by the model works by detecting the most "confident" item. Adversarial patches exploit this by creating inputs that appear more "confident" than the actual object, misleading the image classification model.

# Project Introduction

This project is based on the Adversarial Patch paper, implementing key equations into code and performing adversarial attacks on a pre-trained YOLOv8s-cls single model using the ImageNet dataset.

Implemented using the Pytorch framework, **the purpose of this project is to evaluate how robust and preventive measures the latest image classification models released in 2023 have against these attacks, six years after the 2017 paper was published.**

To achieve this, I **wrote code to generate patches that induce input images to be classified into specific classes**, and also wrote code for **testing** and **real-world application testing**. Using the YOLOv8s-cls model and the ImageNet dataset, I trained the patches and ultimately aimed to verify their effectiveness **even in the real world**.

- By applying the concept of **Adversarial Patch introduced in 2017** to the **YOLOv8s-cls model of 2023**, I will show that the **past research is still valid when combined with the latest models**.
- The purpose is to emphasize how severely this technology can be **exploited** by proving that adversarial patches are **applicable in the real world**.

Upon completion of the project, I conducted two experiments where the model classified images as **Orange** and **Toaster**. Through these experiments, I aim to demonstrate how vulnerable image classification models are to these adversarial attacks.

## Key Parts of Patch Training

### Patch Training Step

```python
def train_step(model, images, target_class, device, patch, optimizer):
    train_loss = 0
    train_success = 0
    batch_size = images.size(0)

    for image in images:
        image_np = (image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
        image = preprocess_image(image_np).unsqueeze(0).to(device)
        current_patch = patch.clone()

        # Random transformation
        angle, scale = random_transformation()
        transformed_patch = transform_patch(current_patch, angle, scale, device, "default")

        # Apply patch at random location
        x = torch.randint(0, image.shape[2] - transformed_patch.shape[2], (1,), device=device).item()
        y = torch.randint(0, image.shape[3] - transformed_patch.shape[3], (1,), device=device).item()
        patched_image = apply_patch_to_image(image, transformed_patch, x, y)

        # Model prediction
        result = model(patched_image, verbose=False)

        log_probs = torch.log(result[0].probs.data)
        target_prob = log_probs.unsqueeze(0)
        loss = torch.nn.functional.nll_loss(target_prob, torch.tensor([target_class], device=device))
        success = calculate_success(target_prob, target_class)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Clamp patch values between 0 and 1
        with torch.no_grad():
            patch.data = torch.clamp(patch, 0, 1)

        if result[0].probs.top1 == target_class:
            # Display image
            patched_image_np = (patched_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
            patched_image_np = cv.cvtColor(patched_image_np, cv.COLOR_RGB2BGR)
            cv.imshow("patched_image", patched_image_np)
            cv.waitKey(1)

        train_loss += loss.item()
        train_success += success

    train_loss /= batch_size
    train_success /= batch_size

    return train_loss, train_success
```

- A block of codes related to preprocessing, patch transformation and application of images among the entire code, and model inference.
- More detailed explanations are below.

### Transformation and Application

```python
angle, scale = random_transformation()
transformed_patch = transform_patch(current_patch, angle, scale, device, "default")
x = torch.randint(0, image.shape[2] - transformed_patch.shape[2], (1,), device=device).item()
y = torch.randint(0, image.shape[3] - transformed_patch.shape[3], (1,), device=device).item()
patched_image = apply_patch_to_image(image, transformed_patch, x, y)
```

- Random transformations (rotation, scale) are applied to the patch.
- The transformed patch is applied at a random location on the image.

### Model Prediction and Loss Calculation

```python
result = model(patched_image, verbose=False)
log_probs = torch.log(result[0].probs.data)
target_prob = log_probs.unsqueeze(0)
loss = torch.nn.functional.nll_loss(target_prob, torch.tensor([target_class], device=device))
success = calculate_success(target_prob, target_class)
```

- The patched image is predicted by the model.
- Log probabilities are calculated to compute the loss.

### Patch Update

```python
if optimizer is not None:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

- The patch is updated using the Adam optimizer.

## Experiment

- Patch Creation
    - 40,000 images randomly selected from the train images of the Imagenet Dataset were divided into training and validation sets in an 8:2 ratio.
    - Experiments were conducted with an Orange Patch size of 80 pixels and a Toaster Patch size of 64 pixels. When synthesizing the image, the size was randomly adjusted from 0.7 times to 1.2 times, and rotated randomly by 0 degrees, 90 degrees, 180 degrees, and 270 degrees.
    - The image size was 224 pixels, and the image was converted to a Tensor after synthesizing with the patch, then input to the model.
- Training Process
    - The following is a visual representation of the training process.
        - Training process chart
        - How the patch is synthesized into the image as it is updated

        ![ezgif-6-2dc68b71b6](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/4fe7eedb-6848-46d7-8106-e978425e81d7)

        - How the patch is updated
            - Orange Patch
              
              ![patch](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/30918e5c-d03a-4408-ac69-6647ed40079d)
              
            - Toaster Patch

              ![ezgif-1-21f4f5cdce](https://github.com/seokkkkkk/Adversarial-Patch-Attack-Implemetaion-YOLOv8/assets/66684504/89d897b4-aec9-4e16-a127-7e5b51b14377)

        - Completed Patch
            - Orange Patch
                
                ![49](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/91308919-8fe7-4384-8f9c-3fed4b0c8b6e)

            - Toaster Patch

              ![16](https://github.com/seokkkkkk/Adversarial-Patch-Attack-Implemetaion-YOLOv8/assets/66684504/60cd8ce1-7646-41f5-95de-1826c3718a1c)
            
##Patch Verification

**The following process can be done directly from patch_tester.py.**

```python
import patch
import utils
import ultralytics
import dataset
import numpy as np
import cv2 as cv
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLOv8n-cls 모델 로드
model = ultralytics.YOLO("yolov8s-cls.pt").to(device)

# 테스트 Dataset 폴더 경로
image_path = "C:\\Users\\HOME\\Desktop\\imagenet\\ILSVRC\\Data\\CLS-LOC\\test"

image_path = utils.return_path_to_images(image_path)

# 이미지 폴더 내의 모든 이미지 파일 처리
test_loader = DataLoader(dataset.ImageDataset(image_path, device), batch_size=1, shuffle=False, num_workers=0)

test_patch = patch.patch_init(64, "default", device, "C:\\Users\\HOME\\IdeaProjects\\adversarial_patch\\model\\patch\\16.png")

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
        if top1_class == 859:
            results_correct += 1
        print(f"Current correct rate: {results_correct / total_length * 100:.2f}% Current Images: {total_length}")
```
- After creating the patch, the same preprocessing process was applied, and tests were conducted on 99,999 images of the Imagenet Dataset test images.
    - Orange Patch(80 Pixel)
        - Attack success rate: Classified 89.40% of images as Orange.
                  
    ![output](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/5d2fa52f-e5b0-4f5e-8571-ec95324ccbf7)
    
    - Toaster Patch(64 Pixel)
        - Attack success rate: Classified 72.98% of images as Orange.

    ![ezgif-1-2167c065ae](https://github.com/seokkkkkk/Adversarial-Patch-Attack-Implemetaion-YOLOv8/assets/66684504/620fd696-e602-4397-a958-3bbdda43b311)

## Real-world Application

**The following process can be done directly by running yolo_cls_viewer.py on videos taken by printing learned patches.**

```python
from utils import prediction_chart
from ultralytics import YOLO
import cv2 as cv
import torch
import os

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLOv8n-cls 모델 로드
model = YOLO('../yolov8s-cls.pt').to(device)

# 입력 동영상 폴더 경로
input_video_folder = "video"

# 출력 동영상 폴더 경로
output_video_folder = os.path.join(input_video_folder, 'output')

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_video_folder):
    os.makedirs(output_video_folder)

# 이미지 사이즈
image_size = (640, 640)
tensor_image_size = (224, 224)

# 입력 동영상 폴더 내의 모든 동영상 파일 처리
for filename in os.listdir(input_video_folder):
    if filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mov') or filename.endswith('.MOV'):
        input_video_path = os.path.join(input_video_folder, filename)
        output_video_path = os.path.join(output_video_folder, f"{os.path.splitext(filename)[0]}_output.avi")

        # 동영상 파일 열기
        cap = cv.VideoCapture(input_video_path)

        # 동영상 파일이 열려있는지 확인
        if not cap.isOpened():
            print(f"Error: Could not open video {filename}.")
            continue

        # 동영상 속성 확인
        fps = int(cap.get(cv.CAP_PROP_FPS))
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # 동영상 파일 저장
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(output_video_path, fourcc, fps, (image_size[0] * 2, image_size[1]))

        # 동영상 프레임 읽기
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # 이미지 크기 조정
            frame = cv.resize(frame, image_size)

            tensor_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            tensor_frame = cv.resize(tensor_frame, tensor_image_size)
            tensor_frame = torch.tensor(tensor_frame).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

            # YOLOv8n-cls 모델 추론
            results = model(tensor_frame)

            # 상위 5개의 예측 클래스와 해당 확률을 추출
            probs = results[0].probs  # 확률 추출
            top5_indices = probs.top5  # 상위 5개의 인덱스
            top5_probs = probs.top5conf.to('cpu').numpy()  # 상위 5개의 확률
            top5_classes = [model.names[i] for i in top5_indices]  # 상위 5개의 클래스 이름

            # 예측 차트 생성
            chart_image = prediction_chart(top5_classes, top5_probs)

            # 차트 이미지 크기 조정 (프레임과 동일하게)
            chart_image = cv.resize(chart_image, image_size)

            # 이미지 결합
            combined_image = cv.hconcat([frame, chart_image])

            # 동영상 파일에 프레임 추가
            out.write(combined_image)

            # 실시간으로 보여주기
            cv.imshow('Combined Frame', combined_image)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # 동영상 파일 닫기
        cap.release()
        out.release()
        cv.destroyAllWindows()

print('동영상 처리 완료')
```

To prove that the adversarial patch created based on the paper is effective in the real world and robust to scale and distortion, patches trained to be recognized as Orange were printed in various sizes and tested.

![IMG_5999_output](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/b45cb7ce-f876-4130-8d1f-c23f74773378)

![IMG_5992_output](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/db62f264-8496-4413-ad2c-6b96b027d633)

![1234](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/044be286-0700-40a5-8630-09a81a5e0b81)

- After testing by attaching patches printed at various sizes and angles to real objects, the model recognized the patched objects as "Orange" with high probability. This demonstrates that adversarial patches are effective not only in digital environments but also in real-world environments.
- The experimental results indicate that maliciously generated adversarial patches can also pose a threat to the real world.
- In this example, experiments were conducted to simply recognize it as Orange, but if traffic signs are not detected or misrecognized, serious consequences may occur.
- The development of robustness and defense techniques for these adversarial examples is likely to be important.

## How to execute
- Create Adversarial Patches: Run main.py
    - You can adjust the size of the patch by adjusting the patch_size variable.
    - patch_shapes is currently implemented only default (square).
    - If you insert the desired image in custom_patch_path, it is used as an initial patch.
    - You can adjust the learning rate using the learning_rate variable.
    - You can adjust the number of learning progresses through the epochs variable.
    - Classify target_class Enter the Imagine Class Id of the target object.
    - If stop_threshold is specified, all epochs are not performed and prematurely shut down.
    - images_path specifies the folder where the data to train is located.

- Test created hostile patches: run patch_tester.py
    - In image_path, specify the dataset folder for testing.

- Check Top5 for video: run yolo_cls_viewer.py
    - Create a video containing the Top5 prediction chart for videos that exist in input_video_folder.

## **Experimental Environment**

- Python: 3.9.19
- PyTorch: 2.2.2
- NumPy: 1.26.4
- OpenCV: 4.9.0
- CUDA: 12.1
- Ultralytics: 2.2.2
    - Important: The **`@smart_inference_mode()`** in ultralytics [predictor.py](http://predictor.py/) should be commented out.
- GPU: GTX 1060 6GB

## Contribution

This project was independently developed based on the referenced paper. Contributions are welcome.

Feel free to submit pull requests.

## References

- Adversarial Patch: https://arxiv.org/abs/1712.09665
- ImageNet Dataset was used for patch training.
