import numpy as np
import os
from PIL import Image
from skimage.transform import resize
import sys
from keras.utils import to_categorical
from keras import backend as K
import keras
from keras.models import Model
from keras.models import load_model

def read_file(test):
    folders = os.listdir(test)
    data = []
    labels = []
    for i in range(len(folders)):
        print(test+folders[i])
        images = os.listdir(test+"/"+folders[i])
        for img in images:
            # image = Image.open( "sub_imagenet/train/n02037110/n02037110_18.JPEG")
            image = Image.open(test+"/"+folders[i]+"/"+img)
            image = np.array(image, dtype='uint8')
            data.append(resize(image, (224,224,3)))
            labels.append(i)
        print(i)
        print(np.shape(data))
        # print(np.shape(labels))
    np.save("val_data", data)
    np.save("val_labels", labels)

read_file(sys.argv[1])
classes=10

model_file = sys.argv[2]
model = load_model(model_file)


X_test = np.load("val_data.npy")
Y_test = np.load("val_labels.npy")

Y_test = to_categorical(Y_test, classes)

def score():
    score = model.evaluate(X_test,Y_test)
    print(model.metrics_names[1], score[1]*100)

score()