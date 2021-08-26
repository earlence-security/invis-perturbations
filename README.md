# Invisible Perturbations: Physical Adversarial Examples Exploiting the Rolling Shutter Effect
This repository contains the code and implementation for the paper [Invisible Perturbations: Physical Adversarial Examples Exploiting the Rolling Shutter Effect](https://arxiv.org/abs/2011.13375).

In this work, we propose a new class of physical adversarial examples that are invisible to human eyes. Rather than modifying the victim object with visible artifacts, we modify light that illuminates the object. We demonstrate how an attacker can craft a modulated light signal that adversarially illuminates a scene and causes targeted misclassifications on a state-of-the-art ImageNet deep learning model. Concretely, we exploit the radiometric rolling shutter effect in commodity cameras to create precise striping patterns that appear on images. To human eyes, it appears like the object is illuminated, but the camera creates an image with stripes that will cause ML models to output the attacker-desired classification.

# Generating Transformed Images
Prior to generating the attack signal, we need data augmentation to account for viewpoint and lighting changes in the scene. We provide ```generate_augmented_images.py``` to generate images by applying brightness, rotation and scale transformations. You will need to provide two images of the scene (with only the ambient lighting, and with ambient and attacker controlled lighting at full illumination). You can refer to the sample input images in the ```captured_images``` folder. You can also specify the bounds for different transformation parameters. Appropriate numbers for the parameters have been set as default arguments.
```python
python generate_augmented_images.py --full_illum_image_path ./captured_images/full_illum_image.png --amb_illum_image_path ./captured_images/amb_illum_image.png --num_samples 200 --out_dir ./augmented_images/
```

# Generating the Attack Signal
The script ```generate_attack_signal.py``` generates the attack signal. Currently, it requires a GPU which is suggested as a large number of iterations are needed for a successful physical attack (however, minor changes should also allow it to run without a GPU). 


# Citation
```
@InProceedings{Sayles_2021_CVPR,
    author    = {Sayles, Athena and Hooda, Ashish and Gupta, Mohit and Chatterjee, Rahul and Fernandes, Earlence},
    title     = {Invisible Perturbations: Physical Adversarial Examples Exploiting the Rolling Shutter Effect},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {14666-14675}
}
```
