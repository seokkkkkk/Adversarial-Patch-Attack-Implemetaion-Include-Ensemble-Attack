
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

## Project Introduction

This project is based on the Adversarial Patch paper and implements a single model attack on the Imagenet-pretrained Yolov8 cls model.

Two experiments are designed to classify objects as "orange", "toaster".

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
         
        - Completed Patch
            - Orange Patch
                
                ![49](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/91308919-8fe7-4384-8f9c-3fed4b0c8b6e)

            - Toaster Patch
            
- Patch Verification
    - After creating the patch, the same preprocessing process was applied, and tests were conducted on 99,999 images of the Imagenet Dataset test images.
        - Orange Patch(80 Pixel)
            - Attack success rate: Classified 89.40% of images as Orange.
                      
        ![output](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/5d2fa52f-e5b0-4f5e-8571-ec95324ccbf7)
        
        - Toaster Patch(64 Pixel)
            - Attack success rate:

## Real-world Application
To prove that the adversarial patch created based on the paper is effective in the real world and robust to scale and distortion, patches trained to be recognized as Orange were printed in various sizes and tested.
![IMG_5999_output](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/b45cb7ce-f876-4130-8d1f-c23f74773378)
![IMG_5992_output](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/db62f264-8496-4413-ad2c-6b96b027d633)
![1234](https://github.com/seokkkkkk/adversarial_patch_implementation/assets/66684504/044be286-0700-40a5-8630-09a81a5e0b81)

- The model recognized objects with attached patches as 'Orange' with high probability when patches printed in various sizes and angles were attached to actual objects. This shows that adversarial patches are effective not only in digital environments but also in real environments.
- These experimental results indicate that adversarial patches can pose a threat in the real world. Developing robustness and defense techniques against such adversarial examples is considered important.


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
