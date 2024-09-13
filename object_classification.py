import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))


# model.save('image_classification.keras')

model = models.load_model('image_classification.keras')

# Load and preprocess the image
# img = cv.imread('car.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# img = cv.resize(img, (32, 32))
#
# prediction = model.predict(np.array([img])/255)
# index = np.argmax(prediction)
# print((f'prediction is {class_name[index]}'))
# plt.imshow(img)
# plt.show()

img2 = cv.imread('car.jpg')
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
img2 = cv.resize(img2, (32, 32))

prediction = model.predict(np.array([img2])/255)
index = np.argmax(prediction)
print((f'prediction is {class_name[index]}'))

plt.imshow(img2)
plt.show()