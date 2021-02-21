import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

train_x = mnist.train.images
train_x_pic = np.reshape(train_x[1], (28, 28))
plt.imshow(train_x_pic)
plt.show()