# Install required packages and setup Kaggle API
# !pip install kaggle
# !mkdir -p .kaggle
# !cp kaggle.json .kaggle/
# !chmod 600 .kaggle/kaggle.json

# Download and extract the CIFAR-10 dataset
# !kaggle competitions download -c cifar-10
from zipfile import ZipFile
with ZipFile('cifar-10.zip', 'r') as zip:
    zip.extractall()
# !pip install py7zr
import py7zr
archive = py7zr.SevenZipFile('train.7z', mode='r')
archive.extractall(path='Training Data')
archive.close()

# Import dependencies
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Prepare file and labels
filenames = os.listdir('train')
labelsdf = pd.read_csv('trainLabels.csv')
labelsdictionary = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
labels = [labelsdictionary[i] for i in labelsdf['label']]
idlist = list(labelsdf['id'])

# Load images into numpy arrays
data = []
for id in idlist:
    image = Image.open(f"train/{id}.png")
    image = np.array(image)
    data.append(image)
X = np.array(data)
Y = np.array(labels)

# Train-test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=2)
Xtrainscaled = Xtrain / 255.0
Xtestscaled = Xtest / 255.0

# Import TensorFlow and Keras for the model
import tensorflow as tf
from tensorflow.keras import Sequential, models, layers, optimizers
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50

# Build ResNet50-based classifier
numofclasses = 10
convolutionalbase = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
model = models.Sequential()
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(convolutionalbase)
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(numofclasses, activation='softmax'))

# Compile and train
model.compile(
    optimizer=optimizers.RMSprop(learning_rate=2e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
history = model.fit(Xtrainscaled, Ytrain, validation_split=0.1, epochs=10)

# Evaluate model
loss, accuracy = model.evaluate(Xtestscaled, Ytest)
print("Test Accuracy", accuracy)

# Plot loss and accuracy curves
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

plt.plot(history.history['acc'], label='train accuracy')
plt.plot(history.history['val_acc'], label='validation accuracy')
plt.legend()
plt.show()
