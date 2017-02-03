import os
import cv2
from constants import *
from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers import Conv2D, Dense, Activation, Dropout, merge, Input

model = Sequential()

# CNN BLOCK
images = []
path = './data/img'

for image in os.listdir(path):
    img = cv2.imread(path + image)
    img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)


def build_cnn_block(images, model, batch_size=128, seq_len=10, left_context=5):
    """

    :param images:
    :param model:
    :param batch_size:
    :param seq_len:
    :param left_context:
    :return:
    """
    frame_block = Input(shape=[batch_size, seq_len + left_context, WIDTH, HEIGHT, CHANNELS])

    model.add(Conv2D(32, 3, 3, input_shape=frame_block, border_mode='valid', subsample=(6, 6), name="conv1_3"))
    conv1 = model.add(Dropout(KEEP_PROP))
    aux1 = Dense(128)(conv1)

    model.add(Conv2D(64, 5, 5, border_mode='valid', name="conv2_5"))
    conv2 = model.add(Dropout(KEEP_PROP))
    aux2 = Dense(128)(conv2)

    model.add(Conv2D(64, 5, 5, border_mode='valid', name="conv3_5"))
    conv3 = model.add(Dropout(KEEP_PROP))
    aux3 = Dense(128)(conv3)

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(KEEP_PROP))

    model.add(Dense(256))
    model.add(Activation('relu'))
    fl2 = model.add(Dropout(KEEP_PROP))

    # Merge residual layers
    model.add(merge([fl2, aux1, aux2, aux3], mode='concat'))

    return model

model = build_cnn_block(images, model, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, left_context=LEFT_CONTEXT)
# Simple GRU Architecture
model.add(GRU(input_shape=(None, 128), output_dim=128, return_sequences=True))
model.add(GRU(output_dim=OUTPUT_DIM, return_sequences=False))


# First GRU Layer
# model.add(GRU(output_dim=128, dropout_U=KEEP_PROP, dropout_W=KEEP_PROP))
# Output layer




