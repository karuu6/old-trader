from sklearn.model_selection import train_test_split
from extract import extract_features, label_samples
from tensorflow import keras
import tensorflow as tf
import numpy as np
from conf import *
import pickle
import sys


features = np.array(pickle.load(open('features', 'rb')))
prices = np.array(pickle.load(open('prices', 'rb')))
i_size = 450
n_classes = 3

feats, Y = label_samples(features, prices)

X_train, X_test, y_train, y_test = train_test_split(feats, Y, test_size=0.2)

model = keras.Sequential([
	keras.layers.Dense(i_size, activation='tanh'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='tanh'),
    keras.layers.Dense(64,  activation='relu'),
    keras.layers.Dense(32,  activation='tanh'),
    keras.layers.Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=EPOCHS)

test_loss, test_acc = model.evaluate(X_test,  y_test)
print('\nTest accuracy:', test_acc)

keras.models.save_model(model, MODEL_DIR+'nog')