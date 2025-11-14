🐶🐱 Dog vs Cat Image Classification using Transfer Learning (Project Description)

This project builds an image classification system that distinguishes between dogs and cats using transfer learning with a pretrained MobileNetV2 model from TensorFlow Hub. The goal is to leverage a lightweight yet powerful deep learning model trained on the large-scale ImageNet dataset and adapt it to a binary classification task using a smaller, custom dataset.

The workflow begins by loading the raw dataset of dog and cat images, resizing them to a uniform resolution of 224 × 224, and converting them into NumPy arrays suitable for training. Images are labeled based on their filename patterns: cat images are assigned label 0, and dog images are assigned label 1.

After preprocessing, the dataset is normalized and split into training (80%) and testing (20%) subsets. Instead of training a convolutional neural network from scratch, the project uses MobileNetV2 (Feature Vector) as a fixed feature extractor. This model, pretrained on over a million images, provides high-level visual features that significantly reduce training time while improving accuracy on small datasets.

A simple classification head—a fully connected dense layer with 2 output neurons—is attached on top of the feature extractor. The model is trained for 5 epochs using the Adam optimizer and sparse categorical cross-entropy loss.

Once trained, the model is evaluated on the test set to measure accuracy and generalization capability. The project also includes an interactive prediction module where the user can input the path of any dog or cat image. The image is resized, normalized, passed through the model, and the predicted label (Dog/Cat) is displayed.

Overall, this project demonstrates the effectiveness of transfer learning, feature extraction, and modern deep learning workflows for small-scale image classification problems. It highlights how pretrained CNN architectures can be adapted to specialized tasks with minimal computation while maintaining strong performance.