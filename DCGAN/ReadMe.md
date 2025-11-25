MNIST GAN — Deep Convolutional Generative Adversarial Network

This project implements a DCGAN (Deep Convolutional GAN) that generates handwritten digit images similar to the MNIST dataset.
It follows the original GAN architecture proposed by Goodfellow et al. and uses a convolutional generator–discriminator framework.

🚀 Project Overview

GANs (Generative Adversarial Networks) consist of two models:

Generator – Creates fake images from random noise.

Discriminator – Classifies images as real or fake.

Both networks compete:

Generator tries to fool the discriminator.

Discriminator tries to catch fake samples.

This adversarial training allows the generator to learn how to create realistic MNIST-like digits.

📚 Architecture Summary
Generator

Input: 100-dim noise vector

Dense layer → 7×7×256 feature map

Conv2DTranspose layers for upsampling

Outputs a 28×28×1 grayscale image

Activation: tanh

Discriminator

Convolutional layers (stride 2)

LeakyReLU activations

Dropout for regularization

Final Dense layer outputs a real/fake logit

Loss: Binary Cross Entropy
Optimizers: Adam (1e-4)

📁 Folder Structure
├── training_checkpoints/    
├── image_at_epoch_0001.png  
├── image_at_epoch_0050.png  
├── image_at_epoch_0100.png  
├── README.md  
└── mnist_gan.ipynb / script.py