import os
import numpy as np
import cv2
import glob
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

image_directory = '/content/image resized/'
image_extensions = ['jpg', 'png']

files = []
[files.extend(glob.glob(image_directory + '*.' + e)) for e in image_extensions]

dog_cat_images = np.asarray([cv2.imread(file) for file in files])

filenames = os.listdir(image_directory)
labels = []

for filename in filenames[:len(dog_cat_images)]:
    label = filename[:3]
    if label == 'dog':
        labels.append(1)
    else:
        labels.append(0)

X = dog_cat_images
Y = np.asarray(labels)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

X_train = X_train / 255.0
X_test = X_test / 255.0

mobilenet_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
hub_layer = hub.KerasLayer(mobilenet_url, trainable=False)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = hub_layer(inputs)
outputs = tf.keras.layers.Dense(2)(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(X_train, Y_train, epochs=5)
loss, acc = model.evaluate(X_test, Y_test)
print(loss, acc)

input_image_path = input('Path of the image to be predicted: ')
input_image = cv2.imread(input_image_path)
input_image_resize = cv2.resize(input_image, (224,224))
input_image_scaled = input_image_resize / 255.0
image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])

prediction = model.predict(image_reshaped)
pred_label = np.argmax(prediction)

if pred_label == 0:
    print("Cat")
else:
    print("Dog")
