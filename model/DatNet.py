from keras.layers import Input, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D, merge, Lambda
from keras.layers.convolutional import Cropping2D
from keras.layers import TimeDistributed, GRU, Embedding
from keras.models import Model
from keras.objectives import mean_squared_error
from FLAGS import *
import numpy as np


class DatNet(object):
    def __init__(self):
        self.vision_model = None
        self.RNN = self.build()

    def build(self):
        #
        # INPUT LAYERS
        # ###################################################################################
        frame = Input(shape=(HEIGHT, WIDTH, CHANNELS))
        frame_sequence = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, HEIGHT, WIDTH, CHANNELS))

        # VISION MODEL - USING CNN
        # ####################################################################################
        net = Lambda(lambda image: image/255.0 - 0.5, input_shape=(HEIGHT, WIDTH, CHANNELS))(frame)
        net = Convolution2D(32, 3, 3, init=INIT, activation='relu', border_mode='same')(net)
        net = MaxPooling2D((2, 2), strides=(2, 2))(net)
        net = Dropout(KEEP_PROP)(net)
        ax1 = Dense(256)(Flatten()(net))  # Residual Layer - Like ResNet - To improve gradient flow

        net = Convolution2D(64, 3, 3, init=INIT, activation='relu', border_mode='same')(net)
        net = MaxPooling2D((2, 2), strides=(2, 2))(net)
        net = Dropout(KEEP_PROP)(net)
        ax2 = Dense(256)(Flatten()(net))  # Residual Layer - Like ResNet

        net = Convolution2D(64, 3, 3, init=INIT, activation='relu', border_mode='same')(net)
        net = MaxPooling2D((2, 2), strides=(2, 2))(net)
        net = Dropout(KEEP_PROP)(net)
        net = Flatten()(net)
        ax3 = Dense(256)(net)             # Residual Layer - Like ResNet

        net = Dense(512, init=INIT, activation='relu')(net)
        net = Dropout(KEEP_PROP)(net)
        net = Dense(256, init=INIT, activation='relu')(net)
        net = Dropout(KEEP_PROP)(net)
        net = merge([net, ax1, ax2, ax3], mode='sum')
        self.vision_model = Model(input=frame, output=net)
        # # RNN MODEL, STACKED ON TOP OF THE CNN
        # # ###################################################################################
        net = TimeDistributed(self.vision_model)(frame_sequence)
        net = GRU(HIDDEN_UNITS, return_sequences=True, stateful=True)(net)
        net = TimeDistributed(Dense(128, activation='relu'))(net)
        net = TimeDistributed(Dense(1))(net)
        model = Model(input=frame_sequence, output=net)
        return model

    def train(self, batch_generator=None, epochs=2, augmentation_scale=3):
        if batch_generator is None:
            print("Cannot open batch generator. Please try again.")
        elif self.RNN is None:
            print("RNN is not built yet.")
        else:
            print('Train...')
            for epoch in range(epochs):
                mean_tr_acc = []
                mean_tr_loss = []
                batch_len = 0
                print("Epoch: ", epoch + 1)
                limit = batch_generator.get_seq_size()
                while batch_generator.cursor < limit:
                    # Data Augmentation per patch
                    batch_x, batch_y = batch_generator.next(augmentation_scale)
                    batch_len += len(batch_x)
                    steps = int(len(batch_x) / BATCH_SIZE)
                    # Iterate through batch. (BATCH, TIME STEPS, HEIGHT, WIDTH, CHANNELS)
                    for i in range(0, steps, BATCH_SIZE):
                        x = batch_x[i:i + BATCH_SIZE]
                        y = batch_y[i:i + BATCH_SIZE]
                        # Train on batch, gradient descent will flow back through batch
                        loss, acc = self.RNN.train_on_batch(x, y)
                        mean_tr_acc.append(acc)
                        mean_tr_loss.append(loss)
                        # Reset the state because my RNN is stateful
                        self.RNN.reset_states()
                print('Accuracy: {}, Loss: {}'.format(np.mean(mean_tr_acc), np.mean(mean_tr_loss)))
                print('Batches ', batch_len)
                print('_____________________________________________________________')
                batch_generator.reset_cursor()

    def load_weights(self, saved_weights):

        self.RNN.load_weights(saved_weights)

    def model(self):
        return self.RNN


def mse_steer_angle(y_true, y_pred):
    return mean_squared_error(y_true[0], y_pred[0])