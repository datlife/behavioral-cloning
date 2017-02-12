import pickle
import cv2
import numpy as np


# ########## FILE READER ##############
# #####################################
data = pickle.load(open('./train.p', 'rb'))
images = data['features']
measurements = data['labels']


# #############################
# ## DATA AUGMENTATION ########
###############################
from utils.image_processor import random_transform
import numpy as np
augmented_images = []
augmented_measurements = []

for image, angle in zip(images, measurements):
    flipped_image = cv2.flip(image, 1)
    flipped_angle = float(angle[0]) * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_angle)
    rand_image, _ = random_transform(image)
    augmented_images.append(rand_image)
    augmented_measurements.append(float(angle[0]))

# TODO:
# How to divide this into frame block for RNN
print(np.shape(images))
print(np.shape(augmented_images))
images = np.concatenate((images, augmented_images))
measurements = np.concatenate((np.transpose(measurements)[0], augmented_measurements))

X_train = np.array(images, dtype='uint8')
y_train = np.array(measurements)

# ########## MODEL #################
# ##################################
import json
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from FLAGS import *
from resnet import ResNetPreAct
print(X_train.shape)
print(y_train.shape)


# INPUT LAYERS
# ###################################################################################
frame_sequence = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, HEIGHT, WIDTH, CHANNELS))
# ResNet
# ####################################################################################
model = ResNetPreAct(input_shape=(HEIGHT, WIDTH, CHANNELS), res_layer_params=(3, 16, 4), nb_classes=1)

# Train
model.compile(optimizer=Adam(lr=LEARN_RATE), loss='mse', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=EPOCHS, batch_size=128, verbose=1)

# Save model
json_string = model.to_json()
with open('rnn2.json', 'w') as outfile:
    outfile.write(json_string)

model.save_weights('rnn2.h5')
print('Model saved')
