import pickle
import numpy as np

from FLAGS import *                                   # Stores parameters and hy
from keras.optimizers import Adam
from model.BatchGenerator import BatchGenerator       # Generate batch file to feed into my RNN
from model.DatNet import DatNet, mse_steer_angle      # Compute the loss of steering angle
import matplotlib.pyplot as plt
plt.interactive(False)

# Import data
data = pickle.load(open('./data/train.p', 'rb'))
X_train = data['features']
y_train = data['labels']
y_train = np.expand_dims(np.transpose(y_train)[0], axis=1)
print(np.shape(X_train), np.shape(y_train))

# Build new model
rnn = DatNet()
# rnn.load_weights('model.h5')

# Compile
rnn.model().compile(optimizer=Adam(lr=0.001), loss='mse',
                    metrics=['accuracy'])   # I want to make sure that RNN optimizes for steering angle

# Batch Generator that will augment data and divide original training set into batches
batch_gen = BatchGenerator(sequence=X_train, labels=y_train, seq_len=TIME_STEPS, batch_size=BATCH_SIZE)

# Train model
# rnn.model().fit(X_train, y_train, nb_epoch=3, validation_split=0.2)
rnn.train(batch_gen, epochs=EPOCHS, augmentation_scale=2)

# Save model
json_string = rnn.model().to_json()
with open('model.json', 'w') as outfile:
    outfile.write(json_string)

rnn.model().save_weights('model.h5')
print('Model saved')

#
# # Post-process angle
# #








