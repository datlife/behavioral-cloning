from keras.models import Model
from keras.layers import Input, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D, Embedding, merge
# from keras.layers import TimeDistributed
# from keras.layers.recurrent import GRU

# INPUT IMAGE SIZE
HEIGHT = 160
WIDTH = 320
CHANNELS = 3


class Model(object):

    def __init__(self, trainable=True):
        self.train = trainable
        self.model = None

    def build_cnn(self):
        # CNN Model - Pre-trained Model
        features = Input(shape=(HEIGHT, WIDTH*3, CHANNELS))
        x = Convolution2D(3, 1, 1, border_mode='same')(features)
        x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
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
        y = Dropout(.5)(y)      # <-- Trim last layer and feed to GRU RNN - Need to te trained first
        predictions = Dense(2)(y)

        cnn_model = Model(input=features, ouput=predictions)

        return cnn_model

# RNN Model


# Post-Processing Steering Angles from Model
