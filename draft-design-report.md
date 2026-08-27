# FGSM / PGD Adversarial Examples on an Image Classifier

## Background: The Vulnerability of Deep Neural Networks

Deep Neural Networks (DNNs) achieve high performance on image classification tasks, but they are surprisingly fragile. By adding imperceptible, carefully crafted perturbations to an input image, an attacker can trick a high-accuracy classifier into outputting an incorrect prediction with high confidence. These modified inputs are called **adversarial examples**.