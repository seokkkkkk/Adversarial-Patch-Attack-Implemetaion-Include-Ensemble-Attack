# 적대적 패치
[English](./README_en.md)
## 소개

적대적 패치는 2017년 Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer 등의 논문 “Adversarial Patch”에서 처음 소개된 객체 인식 모델 공격 기법입니다. 이 기법은 이미지의 특정 부분에 패치를 삽입하여 객체 인식 모델이 객체를 인식하지 못하거나 다른 객체로 인식하도록 만듭니다.

## 수식 설명
<img width="451" alt="Untitled" src="https://github.com/seokkkkkk/yolov8n-cls_adversarial_patch/assets/66684504/754a20ac-9905-4b2d-94d8-fb0293deb6d5">


- 수식의 목적은 패치 p가 이미지 x의 어떤 위치 l에, 특정 변환 t를 거쳐 적용되었을 때, 분류기 Pr이 해당 이미지를 목표 클래스 y^로 인식할 확률을 최대화하는 것입니다.
- 이는 다양한 이미지 x, 변환 t, 위치 l에 대하여 기댓값의 average를 최적화 하는 방식으로 이루어집니다.

## 훈련 과정

- 패치 p는 다양한 이미지 x, 변환 t, 위치 l을 기반으로 반복적으로 학습됩니다.
    - 학습 과정에서 패치 p는 경사 하강법을 통해 목표 클래스 y^로 분류될 확률을 최대화하는 방향으로 업데이트됩니다.

## 원리

- 분류 모델의 이미지 분류는 가장 “확실한” 항목을 감지하는 방식으로 이루어집니다. 적대적 패치는 실제 객체보다 더 “확실한” 방향으로 입력을 생성하여 이미지 분류 모델의 원리를 악용합니다.

## 프로젝트 소개

이 프로젝트는 적대적 패치 논문을 기반으로, Imagenet으로 pretrained된 Yolov8의 cls 모델에 대한 단일 모델 공격을 수행하는 코드를 작성한 것입니다.

모델이 객체를 "orange"로 분류하도록 설계되었습니다.

## 패치 학습 핵심 부분

### 패치 학습 단계

```python
def train_step(model, images, target_class, device, patch, optimizer):
    train_loss = 0
    train_success = 0
    batch_size = images.size(0)

    for image in images:
        image_np = (image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
        image = preprocess_image(image_np).unsqueeze(0).to(device)
        current_patch = patch.clone()

        # 랜덤 변환
        angle, scale = random_transformation()
        transformed_patch = transform_patch(current_patch, angle, scale, device, "default")

        # 랜덤 위치에 패치 적용
        x = torch.randint(0, image.shape[2] - transformed_patch.shape[2], (1,), device=device).item()
        y = torch.randint(0, image.shape[3] - transformed_patch.shape[3], (1,), device=device).item()
        patched_image = apply_patch_to_image(image, transformed_patch, x, y)

        # 모델 예측
        result = model(patched_image, verbose=False)

        log_probs = torch.log(result[0].probs.data)
        target_prob = log_probs.unsqueeze(0)
        loss = torch.nn.functional.nll_loss(target_prob, torch.tensor([target_class], device=device))
        success = calculate_success(target_prob, target_class)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 패치 값을 0과 1 사이로 클램핑
        with torch.no_grad():
            patch.data = torch.clamp(patch, 0, 1)

        if result[0].probs.top1 == target_class:
            # 이미지 imshow
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

### 변환 및 적용

```python
angle, scale = random_transformation()
transformed_patch = transform_patch(current_patch, angle, scale, device, "default")
x = torch.randint(0, image.shape[2] - transformed_patch.shape[2], (1,), device=device).item()
y = torch.randint(0, image.shape[3] - transformed_patch.shape[3], (1,), device=device).item()
patched_image = apply_patch_to_image(image, transformed_patch, x, y)
```

- 패치에 랜덤 변환(회전, 스케일)을 적용합니다.
- 변환된 패치를 이미지의 랜덤한 위치에 적용합니다.

### 모델 예측 및 손실 계산

```python
result = model(patched_image, verbose=False)
log_probs = torch.log(result[0].probs.data)
target_prob = log_probs.unsqueeze(0)
loss = torch.nn.functional.nll_loss(target_prob, torch.tensor([target_class], device=device))
success = calculate_success(target_prob, target_class)
```

- 패치가 적용된 이미지를 모델로 예측합니다.
- 로그 확률을 계산하여 손실을 구합니다.

### 패치 업데이트

```python
if optimizer is not None:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

- Adam 옵티마이저를 사용하여 패치를 업데이트합니다.

## 실험 결과

- 작성예정

## **실험 환경**

- Python: 3.9.19
- PyTorch: 2.2.2
- NumPy: 1.26.4
- OpenCV: 4.9.0
- CUDA: 12.1
- Ultralytics: 2.2.2
    - 중요: ultralytics [predictor.py](http://predictor.py/)의 `@smart_inference_mode()`를 주석처리해야 합니다.
- GPU: GTX 1060 6GB

## 기여

이 프로젝트는 논문을 참고하여 독자적으로 작성된 것입니다. 기여를 환영합니다.

풀 리퀘스트를 적극적으로 제출해주세요.

## 참고

- Adversarial Patch : https://arxiv.org/abs/1712.09665
- 패치 학습을 위해 ImageNet Dataset을 사용하였습니다.
