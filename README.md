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

# 프로젝트 소개

이 프로젝트는 Adversarial Patch 논문을 기반으로 주요 수식을 코드로 구현하고, ImageNet 데이터셋으로 사전 학습된 YOLOv8s-cls 단일 모델에 대한 적대적 공격을 수행한 것입니다. Pytorch 프레임워크를 사용하여 구현되었으며, **이 프로젝트의 목적은 2017년 발표된 논문 이후 6년이 지난 시점인 2023년에 공개된, 최신 이미지 분류 모델이 이러한 공격에 대해 얼마나 강건하고 예방 대책을 가지고 있는지 평가하는 것입니다.**

이를 위해, **입력 이미지를 특정 클래스로 분류되도록 유도하는 패치를 생성하는 코드를 작성**하고, **테스트**와 **현실 세계 적용 테스트**를 위한 코드도 작성하였습니다. YOLOv8s-cls 모델과 ImageNet 데이터셋을 사용하여 패치를 학습하고, 최종적으로 **현실 세계에서도 유효함**을 검증하는 것을 목표로 하였습니다. 
- 2017년에 소개된 Adversarial Patch 개념을 2023년의 YOLOv8s-cls 모델에 적용함으로써, 과거의 연구를 최신 모델과 결합하여 여전히 유효함을 보여줄 것입니다.
- 적대적 패치가 **현실 세계에서도 적용 가능함을 증명**하여, 이러한 기술이 얼마나 심각하게 **악용될 가능성**이 있는지를 강조하는 것을 목적으로 합니다.

프로젝트 완성 후, 모델이 이미지를 **Orange**와 **Toaster**로 분류하도록 두 번의 실험을 진행하였습니다. 이를 통해 이미지 분류 모델들이 이러한 적대적 공격에 얼마나 취약한지를 보여주고자 합니다.

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

**다음의 과정은 main.py를 통해 직접 수행할 수 있습니다.**

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
                     
## 패치 검증

**다음의 과정은 patch_tester.py에서 직접 수행할 수 있습니다.**    
    
- 패치 생성 후 동일한 전처리 과정을 거쳐 Imagenet Dataset의 test 이미지 99,999장에 대해 테스트를 진행하였습니다
    - Orange Patch(80 Pixel)
        - 공격 성공률 : 이미지 중 89.40%를 Orange로 분류하였습니다.
        - 이것은 이미지의 6.25%(0.7배) ~ 18.37(1.2배)% 면적을 차지하는 패치가 적대적 공격을 성공적으로 수행함을 보여줍니다.
                  
    ![output](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/5d2fa52f-e5b0-4f5e-8571-ec95324ccbf7)
    
    - Toaster Patch(64 Pixel)
        - 공격 성공률 :

## 실제 적용

**다음의 과정은 학습된 패치를 프린트하여 촬영한 동영상을 yolo_cls_viewer.py를 실행하여 직접 수행할 수 있습니다.**

논문을 기반으로 프로젝트를 진행하여 생성한 적대적 패치가 현실 세계에서도 유효하며, 스케일 및 왜곡에 대한 강건성이 있음을 증명하기 위해 Orange로 인식되도록 훈련된 패치를 다양한 사이즈로 출력하여 테스트를 진행하였습니다. **(이미지 로드에 시간이 소요됩니다)**
![IMG_5999_output](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/b45cb7ce-f876-4130-8d1f-c23f74773378)
![IMG_5992_output](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/db62f264-8496-4413-ad2c-6b96b027d633)
![1234](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/044be286-0700-40a5-8630-09a81a5e0b81)

- 다양한 크기와 각도로 출력된 패치를 실제 사물에 부착하여 테스트한 결과, 모델은 높은 확률로 패치가 부착된 사물을 'Orange'로 인식했습니다. 이는 적대적 패치가 디지털 환경뿐만 아니라 실제 환경에서도 효과적임을 보여줍니다.
- 이 실험 결과는, 악의적으로 생성된 적대적 패치가 현실 세계에도 위협을 초래할 수 있다는 것을 나타내고 있습니다.
- 이 예제에서는 단순히 Orange로 인식되도록 실험을 진행하였지만, 교통 표지판 등을 탐지하지 못하게 하거나 오인식 하게 할 경우 심각한 결과를 초래할 가능성이 있습니다.
- 이러한 적대적 예제에 대한 강건성 및 방어 기법의 개발이 중요할 것으로 보입니다.


## 실행 방법
- 적대적 패치 생성: main.py 실행
    - patch_size 변수를 조절하여 패치의 사이즈를 조절할 수 있습니다.
    - patch_shapes는 현재 default(사각형)만 구현되었습니다.
    - custom_patch_path에 원하는 이미지를 삽입하면, initial patch로 사용됩니다.
    - learning_rate 변수를 통해 학습률을 조정할 수 있습니다.
    - epochs 변수를 통해 학습 진행 횟수를 조절할 수 있습니다.
    - target_class를 분류 목표 객체의 Imagenet Class Id를 입력합니다.
    - stop_threshold를 지정하면 모든 epochs를 수행하지 않고, 조기 종료합니다.
    - images_path는 훈련할 데이터가 있는 폴더를 지정합니다.
  
- 생성된 적대적 패치 테스트: patch_tester.py 실행
    - image_path에 테스트를 위한 데이터셋 폴더를 지정합니다.

- 동영상에 대한 Top5 에측 확인: yolo_cls_viewer.py 실행
    - input_video_folder에 존재하는 동영상에 대해서 Top5 예측 차트를 포함한 동영상을 생성합니다.
 
## 후속 목표
- YOLO 모델 및 다양한 모델을 Cross Training하여, 다양한 모델에 범용적으로 사용될 수 있는 패치 테스트
  - 다양한 모델을 교차 훈련하여, 여러 모델에 대해 효과적으로 작동하는 범용적인 적대적 패치를 생성 및 테스트. 모델 간의 상호 학습과 검증을 통해 패치의 범용성 평가.
- 적대적 패치를 방어할 수 있는 기법 탐구 및 실험

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
