# ResNet50 Image Classification Project

## Overview
This project demonstrates the use of the ResNet50 deep learning architecture for image classification on the CIFAR-10 dataset. The goal is to build a robust classifier that can distinguish between 10 different object classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using transfer learning with a pre-trained ResNet50 model.

## Dataset
- **CIFAR-10**: A widely used benchmark dataset containing 60,000 32x32 color images in 10 classes, with 6,000 images per class.
- The dataset is split into 50,000 training images and 10,000 test images.

## Methodology

### Data Preprocessing
- Images are resized to 224x224 to match the input requirements of ResNet50.
- Pixel values are normalized to the range [0, 1].
- Labels are mapped to integers for training.

### Model Architecture
- **ResNet50**: A deep convolutional neural network with 50 layers, pre-trained on ImageNet.
- The top classification layer is replaced to match the 10 output classes of CIFAR-10.
- Additional layers (BatchNormalization, Dense, Dropout) are added for fine-tuning and regularization.

### Training
- The model is compiled using the RMSprop optimizer and sparse categorical cross-entropy loss.
- Training is performed for a set number of epochs with a validation split to monitor performance.

## Results
- The model achieves high accuracy on the CIFAR-10 test set, demonstrating the effectiveness of transfer learning with ResNet50.
- Training and validation accuracy/loss curves are plotted to visualize learning progress.

## Usage
- The code can be run in a Colab environment or locally with appropriate dataset paths.
- The trained model can be used for inference on new images.

## References
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. arXiv:1512.03385.
- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- ResNet50 documentation: https://keras.io/api/applications/resnet/
