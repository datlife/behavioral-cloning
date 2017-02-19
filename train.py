import pickle
import numpy as np
import cv2
from FLAGS import *                                   # Stores parameters and hy
from keras.optimizers import Adam
from model.DatNet import DatNet, mse_steer_angle      # Compute the loss of steering angle
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
plt.interactive(False)

# Import data
# ########## FILE READER ##############
# #####################################
data = pickle.load(open('./data/recovery_map2.p', 'rb'))
data2 = pickle.load(open('./data/map_2_four_center_driving.p', 'rb'))

images = data['features']
measurements = data['labels']
images2 = data2['features']
measurements2 = data2['labels']

a = len(images)
b = len(images2)

# Get center images only image = concat(center,left,right)
# images = images[:int(a/3)]
# measurements = measurements[:int(a/3)]
images2 = images2[:int(b/3)]
measurements2 = measurements2[:int(b/3)]

images = np.concatenate((images, images2))
measurements = np.concatenate((measurements, measurements2))

print("Data loaded : Input {} // Measurement {}".format(np.shape(images), np.shape(measurements)))

# #############################
# ## DATA AUGMENTATION ########
###############################
from utils.image_processor import random_transform


augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
    flipped_image = cv2.flip(image, 1)
    augmented_images.append(flipped_image)

    flipped_angle = measurement[0] * -1.0
    augmented_measurements.append((flipped_angle, measurement[1], measurement[2]))

    rand_image, _ = random_transform(image)
    augmented_images.append(rand_image)
    augmented_measurements.append(measurement)

# # TODO:
# How to divide this into frame block for RNN
print(np.shape(images))
print(np.shape(augmented_images))
images = np.concatenate((images, augmented_images))
measurements = np.concatenate((measurements, augmented_measurements))


# Build new model
model = DatNet(input_shape=(HEIGHT, WIDTH, CHANNELS), res_layer_params=(3, 32, 4), nb_classes=1)
model.vision_model.load_weights('./model/cnn.h5')
model.vision_model.summary()
print("Pre-trained model loaded...")
model.vision_model.compile(optimizer=Adam(lr=0.00001), loss=[mse_steer_angle])
callback = ModelCheckpoint('./model/weights.{epoch:02d}-{val_loss:.2f}.h5', save_weights_only=True)
model.vision_model.fit(images, measurements, batch_size=256, nb_epoch=3, callbacks=[callback],
                       validation_split=0.1, shuffle=True)

# Save model
json_string = model.vision_model.to_json()
with open('./model/cnn.json', 'w') as outfile:
    outfile.write(json_string)
model.vision_model.save_weights('./model/cnn.h5')
print('Model saved')
# # Post-process angle
