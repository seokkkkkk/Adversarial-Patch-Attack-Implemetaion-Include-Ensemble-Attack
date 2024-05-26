
# Adversarial Patch

## Introduction

Adversarial Patch is an attack technique on object recognition models, first introduced in the 2017 paper “Adversarial Patch” by Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, and Justin Gilmer. This technique involves inserting a patch into a specific part of an image, causing the object recognition model to either fail to recognize the object or misclassify it as a different object.

## Formula Explanation

![Formula](https://github.com/seokkkkkk/yolov8n-cls_adversarial_patch/assets/66684504/754a20ac-9905-4b2d-94d8-fb0293deb6d5)

- The goal of the formula is to maximize the probability that the classifier \( Pr \) recognizes the image with the patch \( p \), applied at a specific location \( l \) and with a certain transformation \( t \), as the target class \( y^ \).
- This is achieved by optimizing the average expectation over various images \( x \), transformations \( t \), and locations \( l \).

## Training Process

- The patch \( p \) is iteratively trained based on various images \( x \), transformations \( t \), and locations \( l \).
    - During the training process, the patch \( p \) is updated via gradient descent to maximize the probability of being classified as the target class \( y^ \).

## Principle

- Image classification by the model works by detecting the most "confident" item. Adversarial patches exploit this by creating inputs that appear more "confident" than the actual object, misleading the image classification model.

## Project Introduction

This project is based on the Adversarial Patch paper and implements a single model attack on the Imagenet-pretrained Yolov8 cls model.

The model is designed to classify objects as "orange".

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

## Experimental Results

- To be added

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
