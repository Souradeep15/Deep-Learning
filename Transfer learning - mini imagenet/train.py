# import keras
import numpy as np
import os
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


nb_classes = 10
def read_file(train):
    folders = os.listdir(train)
    data = []
    labels = []
    for i in range(len(folders)):
        print(train+"/"+folders[i])
        images = os.listdir(train+"/"+folders[i])
        for img in images:
            # image = Image.open( "sub_imagenet/train/n02037110/n02037110_18.JPEG")
            image = Image.open(train+"/"+folders[i]+"/"+img)
            image = np.array(image, dtype='uint8')
            data.append(resize(image, (224,224,3)))
            labels.append(i)
        print(i)
        print(np.shape(data))
        # print(np.shape(labels))
    np.save("data", data)
    np.save("labels", labels)

read_file(sys.argv[1])

X_train = np.load("data.npy")
Y_train = np.load("labels.npy")

Y_train = to_categorical(Y_train, 10)

X_train,Y_train = shuffle(X_train,Y_train, random_state=2)

image_input = Input(shape=(224,224,3))
model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=image_input, pooling=None,classes=nb_classes)

for layer in model.layers[:5]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
print(np.shape(x))
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)


model_final = Model(input = model.input, output = predictions)

# my_model.layers[3].trainable
model_final.save(sys.argv[2])
model_final.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
              metrics=['accuracy'])



def train():
    model_final.fit(X_train, Y_train,
              batch_size=32,
              epochs=10,
              shuffle=True)
    model_final.save(sys.argv[2])

train()
