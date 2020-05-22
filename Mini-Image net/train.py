from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
import numpy as np
import keras	
import sys
from keras import regularizers
from keras.regularizers import l2
from keras.utils import plot_model
from keras import backend as K
K.set_image_dim_ordering('th')

X_train = np.load(sys.argv[1])
# X_test = np.load('image_net/x_test.npy')
Y_train = np.load(sys.argv[2])
# Y_test = np.load('image_net/y_test.npy')

Y_train = to_categorical(Y_train, num_classes=10)


model = Sequential()

#Layer 1
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation="relu", input_shape=(3,112,112), kernel_regularizer=regularizers.l2(0.0005)))#Convo$
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
# model.add(keras.layers.GaussianNoise(0.1))
#Layer 2
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))#Convo$
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
# model.add(keras.layers.GaussianNoise(0.1))
#Layer 3
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))#Convo$
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
# model.add(keras.layers.GaussianNoise(0.1))
#Layer 4
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.50))
# model.add(keras.layers.GaussianNoise(0.1))
#layer 5
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.50))
# model.add(keras.layers.GaussianNoise(0.1))

#layer 6


#Dense layer
model.add(Flatten())# shape equals to [batch_size, 32] 32 is the number of filters
model.add(Dense(10))#Fully connected layer
model.add(Activation('softmax'))

keras.utils.multi_gpu_model(model, gpus=2, cpu_merge=False, cpu_relocation=False)

#opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
# opt = keras.optimizers.SGD(lr=0.0005, decay=1e-6)
# opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)# best one
# opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#model.save('my_model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


def train():
    model.fit(X_train, Y_train,
              batch_size=128,
              epochs=200,
              shuffle=True)
    model.save(sys.argv[3])

train()
# plot_model(model, to_file='model.png')

