# Face Mask Detection using CNN

## Overview
This project builds a convolutional neural network (CNN) to automatically detect whether a person is wearing a face mask or not from images. The model is trained on a dataset of labeled face images and can be used as a core component for real‑time monitoring systems in public places.[web:11][web:15]

## Dataset
- The dataset is organized into three folders: `train`, `val`, and `test`, each containing two subfolders:
  - `with_mask`
  - `without_mask`
- Images are RGB face crops resized to a fixed resolution (typically 128×128) before being fed into the model.[web:19]

## Methodology

### Preprocessing and Augmentation
- Rescaling of pixel values to the range \([0, 1]\).
- Data augmentation on the training set:
  - Random rotations
  - Width and height shifts
  - Shear transformations
  - Zoom
  - Horizontal flipping
- Separate generators are used for training, validation, and test sets to ensure a clean evaluation pipeline.[web:19]

### CNN Architecture
The model is a custom CNN built with Keras/TensorFlow, consisting of:
- Multiple convolution–max‑pooling blocks with increasing filter depth.
- A flatten layer to convert feature maps to a 1D vector.
- Fully connected layers with ReLU activation and dropout for regularization.
- A final dense layer with a sigmoid activation for binary classification (`mask` vs `no_mask`).[web:18][web:19]

### Training Setup
- Loss function: Binary cross‑entropy.
- Optimizer: Adam with a low learning rate (e.g., 1e‑4) for stable convergence.
- Metrics: Accuracy on training and validation sets.
- Callbacks:
  - Model checkpoint to save the best model based on validation accuracy.
  - Early stopping to prevent overfitting by monitoring validation loss.[web:18][web:19]

## Evaluation
- The trained model is evaluated on the held‑out test set using:
  - Overall test accuracy.
  - Confusion matrix to visualize true/false positives and negatives.
  - Classification report with precision, recall, and F1‑score for both classes (`with_mask`, `without_mask`).[web:18][web:19]
- Plots of training/validation accuracy and loss across epochs are used to analyze learning behavior and detect overfitting.[web:18]

## Inference and Usage
- The project includes a utility function to:
  - Load a single image.
  - Apply the same preprocessing pipeline (resize, normalize).
  - Run the model and output whether the person is wearing a mask or not.
- This function can be extended to work on webcam frames or video streams for real‑time mask detection with minimal changes.[web:16][web:19]

## Applications
- Monitoring mask compliance in:
  - Offices and campuses
  - Shopping malls and supermarkets
  - Transit hubs such as airports and train stations
- Integration into existing CCTV or edge devices for automated alerting and analytics.[web:15][web:16]

## Future Work
- Extend to multi‑class classification (e.g., correct mask, incorrect mask, no mask).
- Replace the custom CNN with a lightweight pre‑trained backbone (e.g., MobileNetV2) to enable deployment on edge and mobile devices.[web:12][web:16]
- Add a real‑time video pipeline with bounding boxes and tracking for multiple faces in a frame.[web:13][web:16]
