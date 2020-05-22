import numpy as np
import keras
import sys
import h5py
from keras.models import load_model

x_test  = np.load(sys.argv[1])
y_test = np.load(sys.argv[2])

y_test = keras.utils.to_categorical(y_test, num_classes = 10)

model = load_model(sys.argv[3])
score = model.evaluate(x_test,y_test)
print(model.metrics_names[1], score[1]*100)