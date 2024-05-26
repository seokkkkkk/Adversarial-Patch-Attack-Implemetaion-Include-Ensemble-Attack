# 적대적 패치
[English](./README_en.md)
## 소개

적대적 패치는 2017년 Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer 등의 논문 “Adversarial Patch”에서 처음 소개된 객체 인식 모델 공격 기법입니다. 이 기법은 이미지의 특정 부분에 패치를 삽입하여 객체 인식 모델이 객체를 인식하지 못하거나 다른 객체로 인식하도록 만듭니다.

기존의 적대적 예제들은 이미지에 미세한 노이즈를 삽입하여 공격하는 Perturbation 기법을 사용했습니다. 이러한 기법은 각 이미지 별로 적대적 예제를 생성해야하는 한계가 있었으며, 현실 세계의 물체에 적용하기 어렵다는 한계가 있었습니다. 그러나 적대적 패치는 하나의 패치로 여러 이미지나 현실 세계의 물체들에 대한 공격까지 수행할 수 있다는 점에서 차별점을 갖습니다.

## 수식 설명
<img width="451" alt="Untitled" src="https://github.com/seokkkkkk/yolov8n-cls_adversarial_patch/assets/66684504/754a20ac-9905-4b2d-94d8-fb0293deb6d5">


- 수식의 목적은 패치 p가 이미지 x의 어떤 위치 l에, 특정 변환 t를 거쳐 적용되었을 때, 분류기 Pr이 해당 이미지를 목표 클래스 y^로 인식할 확률을 최대화하는 것입니다.
- 이는 다양한 이미지 x, 변환 t, 위치 l에 대하여 기댓값의 평균을 최적화 하는 방식으로 이루어집니다.

## 훈련 과정

- 패치 p는 다양한 이미지 x, 변환 t, 위치 l을 기반으로 반복적으로 학습됩니다.
    - 학습 과정에서 패치 p는 경사 하강법을 통해 목표 클래스 y^로 분류될 확률을 최대화하는 방향으로 업데이트됩니다.

## 원리

- 분류 모델의 이미지 분류는 가장 “확실한” 항목을 감지하는 방식으로 이루어집니다. 적대적 패치는 실제 객체보다 더 “확실한” 방향으로 입력을 생성하여 이미지 분류 모델의 원리를 악용합니다.

## 프로젝트 소개

이 프로젝트는 적대적 패치 논문을 기반으로, Imagenet으로 pretrained된 Yolov8s의 cls 모델에 대한 단일 모델 공격을 수행하는 코드를 작성한 것입니다.

모델이 객체를 "orange", "toaster"로 분류하도록 두 번의 실험을 진행하였습니다.

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

## 실험

- 패치 생성
    - Imagenet Dataset의 train 이미지 중 40,000장을 랜덤으로 선택하여 8:2 비율로 train set과 validation set을 구성하였습니다.
    - Orange Patch의 크기는 80 픽셀, Toaster Patch의 크기는 64 픽셀로 실험하였으며, 이미지를 합성할 때 랜덤으로 0.7배에서 1.2배까지 크기를 조절하고, 0도, 90도, 180도, 270도로 랜덤 회전시켰습니다.
    - 이미지의 크기는 224 픽셀로, 패치와 합성한 후 Tensor로 변환하여 모델에 입력하였습니다.
- 학습 과정
    - 다음은 학습 과정을 시각화한 자료입니다.
        - 학습 과정 차트
        - 패치가 업데이트 되며 이미지에 합성되는 모습
        
       ![ezgif-6-2dc68b71b6](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/4fe7eedb-6848-46d7-8106-e978425e81d7)


        - 패치가 업데이트 되는 모습
            - Orange Patch
                
                ![patch](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/30918e5c-d03a-4408-ac69-6647ed40079d)

            - Toaster Patch
            
        - 완성된 패치
            - Orange Patch
                
                ![49](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/91308919-8fe7-4384-8f9c-3fed4b0c8b6e)

            - Toaster Patch
            
- 패치 검증
    - 패치 생성 후 동일한 전처리 과정을 거쳐 Imagenet Dataset의 test 이미지 99,999장에 대해 테스트를 진행하였습니다
        - Orange Patch(80 Pixel)
            - 공격 성공률 : 이미지 중 89.40%를 Orange로 분류하였습니다.
                      
        ![output](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/5d2fa52f-e5b0-4f5e-8571-ec95324ccbf7)
        
        - Toaster Patch(64 Pixel)
            - 공격 성공률 :

## 실제 적용
논문을 기반으로 프로젝트를 진행하여 생성한 적대적 패치가 현실 세계에서도 유효하며, 스케일 및 왜곡에 대한 강건성이 있음을 증명하기 위해 Orange로 인식되도록 훈련된 패치를 다양한 사이즈로 출력하여 테스트를 진행하였습니다.
![IMG_5999_output](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/b45cb7ce-f876-4130-8d1f-c23f74773378)
![IMG_5992_output](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/db62f264-8496-4413-ad2c-6b96b027d633)
- 다양한 크기와 각도로 출력된 패치를 실제 사물에 부착하여 테스트한 결과, 모델은 높은 확률로 패치가 부착된 사물을 'Orange'로 인식했습니다. 이는 적대적 패치가 디지털 환경뿐만 아니라 실제 환경에서도 효과적임을 보여줍니다.
- 이 실험 결과는, 적대적 패치가 현실 세계에도 위협을 초래할 수 있다는 것을 나타내고 있습니다. 이러한 적대적 예제에 대한 강건성 및 방어 기법의 개발이 중요할 것으로 보입니다.

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
