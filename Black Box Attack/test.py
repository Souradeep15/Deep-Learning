from keras.datasets import cifar10
from keras.models import load_model
from numpy.random import randn, randint
import keras
import tensorflow as tf
from keras import backend as k
import numpy as np
import sys

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
idx = (np.logical_or(y_train == 0, y_train == 1)).reshape(x_train.shape[0])
x_train = x_train[idx]
y_train = y_train[np.logical_or(y_train == 0, y_train == 1)]
y_train = keras.utils.to_categorical(y_train, 2)

new_model = load_model(sys.argv[1])
x_train2 = x_train[11:,:]
y_train2 = y_train[11:,:]
test_m = np.mean(x_train2, axis=0)
xtrain2 = x_train2 - test_m
grads1 = k.gradients(new_model.output, new_model.input)[0]
s1 = tf.compat.v1.Session()
iterate1 = k.function(new_model.input, [grads1])
grad1 = iterate1(x_train2)
grad1 = x_train2 + 0.0625 * np.sign(grad1)
grad1 = grad1.reshape(x_train2.shape[0],32,32,3)
scores = new_model.evaluate(grad1,y_train2)
print("accuracy: ",scores[1])