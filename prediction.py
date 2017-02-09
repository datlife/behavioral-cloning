import csv
import cv2
import numpy as np

LOG_PATH = './data/recovery/driving_log.csv'
IMG_PATH =  './data/recovery/IMG/'
# ########## FILE READER ##############
# #####################################
lines = []
with open(LOG_PATH) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    center_image = line[0]                        # Center image
    tokens = center_image.split('/')
    filename = tokens[-1]                         # Extract image filename
    image_path = IMG_PATH + filename
    image = cv2.imread(image_path)
    image = image[70:150, :, :]
    images.append(image)
    steering_angle = float(line[3])               # Steering angle
    measurements.append(steering_angle)

for line in lines:
    left_image = line[1]                          # Left image
    tokens = left_image.split('/')
    filename = tokens[-1]                         # Extract image filename
    image_path = IMG_PATH + filename
    image = cv2.imread(image_path)
    image = image[70:150, :, :]
    images.append(image)
    steering_angle = float(line[3]) + 0.15         # Steering angle
    measurements.append(steering_angle)

for line in lines:
    right_image = line[2]                           # Right image
    tokens = right_image.split('/')
    filename = tokens[-1]                           # Extract image filename
    image_path = IMG_PATH + filename
    image = cv2.imread(image_path)
    image = image[70:150, :, :]
    images.append(image)
    steering_angle = float(line[3]) - 0.15         # Steering angle
    measurements.append(steering_angle)

from utils.image_processor import random_transform
augmented_images = []
augmented_measurements = []
for image, angle in zip(images, measurements):
    flipped_image = cv2.flip(image, 1)
    flipped_angle = angle * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_angle)
    rand_image, _ = random_transform(image)
    augmented_images.append(rand_image)
    augmented_measurements.append(angle)

# TODO:
# How to divide this into frame block for RNN
print(np.shape(images))
print(np.shape(augmented_images))
images = np.concatenate((images, augmented_images))
measurements = np.concatenate((measurements, augmented_measurements))

X_train = np.array(images)
y_train = np.array(measurements)

# ########## MODEL ##############
# ##################################
import json
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout, merge, TimeDistributed, Input
from keras.layers import   BatchNormalization, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.recurrent import GRU
from FLAGS import *

print(X_train.shape)
print(y_train.shape)
#
# INPUT LAYERS
# ###################################################################################
frame = Input(shape=(HEIGHT, WIDTH, CHANNELS))
frame_sequence = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, HEIGHT, WIDTH, CHANNELS))

# VISION MODEL - USING CNN
# ####################################################################################
net = Lambda(lambda image: image / 255.0 - 0.5, input_shape=(HEIGHT, WIDTH, CHANNELS))(frame)

net = Convolution2D(16, 3, 3, init=INIT, activation='relu', border_mode='same')(net)
net = Convolution2D(16, 3, 3, init=INIT, activation='relu', border_mode='same')(net)
net = MaxPooling2D()(net)
net = Dropout(0.5)(net)

net = Convolution2D(32, 3, 3, init=INIT, activation='relu', border_mode='same')(net)
net = Convolution2D(32, 3, 3, init=INIT, activation='relu', border_mode='same')(net)
net = MaxPooling2D()(net)
net = Dropout(0.5)(net)

net = Convolution2D(64, 3, 3, init=INIT, activation='relu', border_mode='same')(net)
net = Convolution2D(64, 3, 3, init=INIT, activation='relu', border_mode='same')(net)
net = MaxPooling2D()(net)
net = Dropout(0.5)(net)
net = Flatten()(net)

net = Dense(512, init=INIT, activation='relu')(net)
net = Dropout(KEEP_PROP)(net)
net = Dense(256, init=INIT, activation='relu')(net)
net = Dense(1)(net)

model = Model(input=frame, output=net)
# # # RNN MODEL, STACKED ON TOP OF THE CNN
#
# # ###################################################################################
# rnn = TimeDistributed(model)(frame_sequence)
# rnn = GRU(HIDDEN_UNITS, return_sequences=False, stateful=True)(net)
# rnn = Dense(128, activation='relu')(net)
# rnn = Dense(1)(net)
# rnn = Model(input=frame_sequence, output=net)
#
model.load_weights('rnn.h5')

model.compile(optimizer='Adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=3, batch_size=128)
# from model.BatchGenerator import BatchGenerator
# batch_generator = BatchGenerator(X_train, y_train, batch_size=BATCH_SIZE,seq_len=TIME_STEPS)
# print('Train...')
# for epoch in range(2):
#     mean_tr_acc = []
#     mean_tr_loss = []
#     batch_len = 0
#     print("Epoch: ", epoch + 1)
#     limit = batch_generator.get_seq_size()
#     while batch_generator.cursor < limit:
#         # Data Augmentation per patch
#         batch_x, batch_y = batch_generator.next(3)
#         batch_len += len(batch_x)
#         steps = int(len(batch_x) / BATCH_SIZE)
#         # Iterate through batch. (BATCH, TIME STEPS, HEIGHT, WIDTH, CHANNELS)
#         for i in range(0, steps, BATCH_SIZE):
#             x = batch_x[i:i + BATCH_SIZE]
#             y = batch_y[i:i + BATCH_SIZE]
#             # Train on batch, gradient descent will flow back through batch
#             loss, acc = rnn.train_on_batch(x, y)
#             mean_tr_acc.append(acc)
#             mean_tr_loss.append(loss)
#             # Reset the state because my RNN is stateful
#             rnn.reset_states()
#     print('Accuracy: {}, Loss: {}'.format(np.mean(mean_tr_acc), np.mean(mean_tr_loss)))
#     print('Batches ', batch_len)
#     print('_____________________________________________________________')
#     batch_generator.reset_cursor()

# Save model
json_string = model.to_json()
with open('rnn.json', 'w') as outfile:
    outfile.write(json_string)

model.save_weights('rnn.h5')
print('Model saved')
