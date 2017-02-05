import numpy as np
from utils.DataSet import DataSet
from keras.models import Model
from keras.layers import Input, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
plt.interactive(False)


KEEP_PROP = 0.2
LEARN_RATE = 0.001
EPOCHS = 10

LOG_PATH = './data/driving_log.csv'
IMG_PATH = './data/IMG/'

# Import data
data = DataSet(log_path=LOG_PATH, img_dir_path=IMG_PATH)
X_train, y_train = data.get_train_data()

X_train = X_train/255 - 0.5
# Build Model
# CNN Model - Pre-trained Model
features = Input(shape=(80, 160 * 3, 3))
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(features)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Dropout(0.25)(x)

x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Dropout(0.25)(x)

x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Dropout(0.5)(x)

y = Flatten()(x)
y = Dense(1024, activation='relu')(y)
y = Dropout(.5)(y)  # <-- Trim last layer and feed to GRU RNN - Need to te trained first
predictions = Dense(2)(y)

model = Model(input=features, output=predictions)
# Compile and Train
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=8, nb_epoch=3, validation_split=0.2)

# Post-process angle
