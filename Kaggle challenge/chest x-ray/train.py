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
from keras.applications import inception_resnet_v2
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
    train_data = ImageDataGenerator(
            rescale=1./255)


    val_data = ImageDataGenerator(rescale=1./255)

    train_generator = train_data.flow_from_directory(
            folder+'/train',
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical')

    validation_generator = val_data.flow_from_directory(
            folder+'/val',
            target_size=(128,128),
            batch_size=32,
            class_mode='categorical')
    return train_generator, validation_generator

  
  
  
train_generator, validation_generator = read_data(sys.argv[1])
X_train = train_generator[0]
Y_train = train_generator[1]
image_input = Input(shape=(128,128,3))
model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=image_input, pooling=None,classes=2)

for layer in model.layers:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
#x = Dropout(0.1)(x)
predictions = Dense(2, activation="softmax")(x)

model_final = Model(input = model.input, output = predictions)
# model_final.save(sys.argv[2])

model_final.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.0002, decay=0.000001),
              metrics=['accuracy'])



def train():
    mc = keras.callbacks.ModelCheckpoint(sys.argv[2],
                     monitor='val_acc',
                     save_best_only=True,
                     verbose=1)

    es = keras.callbacks.EarlyStopping(monitor='val_acc',
                   patience=5)
    model_final.fit_generator(train_generator,
              epochs=10,
              callbacks = [mc, es],
              validation_data = validation_generator)
    #model_final.save(sys.argv[2])

train()
