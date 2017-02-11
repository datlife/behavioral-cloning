import pickle
import numpy as np

from FLAGS import *                                   # Stores parameters and hy
from keras.optimizers import Adam
from model.BatchGenerator import BatchGenerator       # Generate batch file to feed into my RNN
from model.DatNet import DatNet, mse_steer_angle      # Compute the loss of steering angle
from keras.layers.core import K
import matplotlib.pyplot as plt
plt.interactive(False)

# Import data
# ########## FILE READER ##############
# #####################################
data = pickle.load(open('./train.p', 'rb'))
images = data['features']
measurements = data['labels']
print("Data loaded : Input {} // Measurement {}".format(np.shape(images), np.shape(measurements)))

# Batch generator
batch_gen = BatchGenerator(sequence=images, labels=measurements, seq_len=TIME_STEPS, batch_size=BATCH_SIZE)

# Build new model
model = DatNet(input_shape=(HEIGHT, WIDTH, CHANNELS), res_layer_params=(3, 16, 4), nb_classes=1)

# Load pre-train ResNet model
model.vision_model.load_weights('model/cnn/cnn.h5')

# Trim the last layer for fine tune
# https://github.com/fchollet/keras/issues/2640
for i in range(2):
    model.vision_model.layers.pop()
model.vision_model.layers[-1].outbound_nodes = []
model.vision_model.outputs = [model.vision_model.layers[-1].output]

# # Compile
# # TODO:
# # Find better loss function
model.RNN.compile(optimizer=Adam(lr=0.001), loss=[mse_steer_angle], metrics=['mse'])
model.train(batch_generator=batch_gen, epochs=1, augmentation_scale=4)


# Save model
json_string = model.RNN.to_json()
with open('model.json', 'w') as outfile:
    outfile.write(json_string)
model.RNN.save_weights('model.h5')
print('Model saved')

#
# # Post-process angle
# #
