# import necessary packages for hand gesture recognition project using Python OpenCV
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model 
import matplotlib.pyplot as plt

# Load the gesture recognizer model
model = load_model('keras_model')

# Load class names
f = open('class.txt', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

img = Image.open('test3.jpg').convert('L')
img = img.resize((320, 120))
data = np.array(img)
data = np.array(data, dtype = 'float32')

plt.imshow(data[:, :])
plt.show()

data = data.reshape(1, 120, 320, 1)

prediction = model.predict(data)
print(prediction)