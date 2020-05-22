from keras.datasets import cifar10
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, SGD
import numpy as np
import keras
import tensorflow as tf
from keras import backend as k
from numpy.random import randn, randint
import sys

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

idx = (np.logical_or(y_train == 0, y_train == 1)).reshape(x_train.shape[0])
x_train = x_train[idx]
y_train = y_train[np.logical_or(y_train == 0, y_train == 1)]
y_train = keras.utils.to_categorical(y_train, 2)

x_trainMean = np.mean(x_train[:10,:], axis=0)
x_train = x_train - x_trainMean

model = load_model('./mlp20node_model.h5')
new_model = Sequential()
new_model.add(Flatten(input_shape = x_train.shape[1:]))
#inputs = keras.layers.Input(shape=x_train.shape[1:])
new_model.add(Dense(100))
new_model.add(Activation('relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(100))
new_model.add(Activation('relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(2, activation='softmax'))
#new_model = Model(inputs=inputs, outputs=outputs)
opt = SGD(lr=0.0001)
new_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

epocs=5
x_train1 = x_train[:10,:]
for epoch in range(5):
      print(x_train1.shape)
      y_train1 = model.predict_classes(x_train1)
      y_train1 = keras.utils.to_categorical(y_train1.reshape(-1, 1), 2)
      epoc_steps = int(x_train1.shape[0]/epocs)
      new_model.fit(x_train1,y_train1, steps_per_epoch=epoc_steps, epochs=epocs)
      grads = k.gradients(new_model.output, new_model.input)[0]
      s = tf.compat.v1.Session()
      iterate = k.function(new_model.input, [grads])
      grad = iterate(x_train1)
      #if randint(0, 2) == 1:
      grad= x_train1 + 0.5*np.sign(grad)
      #else:
        #grad= x_train1 - 0.1*np.sign(grad)
      grad = grad.reshape(x_train1.shape[0],32,32,3)
      
      x_train1 = np.append(x_train1, grad, axis=0)
new_model.save(sys.argv[1])