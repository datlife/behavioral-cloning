from keras.models import Model
from keras.layers import Input, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D, Embedding, merge
# from keras.layers import TimeDistributed
# from keras.layers.recurrent import GRU

# INPUT IMAGE SIZE
HEIGHT = 160
WIDTH = 320
CHANNELS = 3

images = []

# CNN Model - Pre-trained Model
features = Input(shape=(HEIGHT, WIDTH, CHANNELS))

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
# y = Dropout(.5)(y) <-- Trim last layer and feed to GRU RNN - Need to te trained first
# y = Dense(1)(y)
vision_model = Model(input=features, output=y)

# RNN Model


# Post-Processing Steering Angles from Model
