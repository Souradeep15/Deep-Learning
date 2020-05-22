import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
import os
from keras.utils import multi_gpu_model
from PIL import Image
from skimage.transform import resize
import sys
from keras.utils import np_utils, generic_utils, to_categorical
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras import backend as K
import keras
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.models import Model
from sklearn.utils import shuffle
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import optimizers

def read_data(folder):
    folders = os.listdir(folder)
    train_data = ImageDataGenerator(
            rescale=1./255,
            validation_split = 0.2)


    train_generator = train_data.flow_from_directory(
            folder,
            target_size=(100, 100),
            batch_size=32,
            class_mode='categorical',
            subset = 'training',
            )

    validation_generator = train_data.flow_from_directory(
            folder,
            target_size=(100,100),
            batch_size=32,
            class_mode='categorical',
            subset = 'validation')
    return train_generator, validation_generator

  
  
  
train_generator, validation_generator = read_data(sys.argv[1])
X_train = train_generator[0]
Y_train = train_generator[1]
image_input = Input(shape=(100,100,3))
model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=image_input, pooling=None,classes=5)

for layer in model.layers:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(5, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

model_final.save(sys.argv[2])

model_final.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])



def train():
    model_final.fit_generator(train_generator,
              epochs=10,
              validation_data=validation_generator)
    model_final.save(sys.argv[2])

train()
