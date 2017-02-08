import pickle
import numpy as np
from BatchGenerator import BatchGenerator       # Generate batch file to feed into my RNN
from DatNet import DatNet, mse_steer_angle      # Compute the loss of steering angle
from FLAGS import *
from keras.optimizers import Adam
import matplotlib.pyplot as plt
plt.interactive(False)


# Import data
data = pickle.load(open('train.p', 'rb'))
X_train = data['features']
y_train = data['labels']
print(np.shape(X_train), np.shape(y_train))

# Build new model
rnn = DatNet()
rnn.load_weights('model.h5')

# # Compile
# rnn.model().compile(optimizer=Adam(lr=0.001), loss='mse',
#                     metrics=[mse_steer_angle, 'accuracy'])    # I want to make sure that Net optimize for steering angle
#
# # Batch Generator that will augment data and divide original training set into batches
# batch_gen = BatchGenerator(sequence=X_train, labels=y_train, seq_len=TIME_STEPS, batch_size=BATCH_SIZE)
#
# # Train model
# rnn.train(batch_gen, epochs=EPOCHS, augmentation_scale=10)
#
#
# # Save model
# json_string = rnn.model().to_json()
# with open('model.json', 'w') as outfile:
#     outfile.write(json_string)
#
# rnn.model().save_weights('model.h5')
print('Model saved')


# Post-process angle
#


import time
from utils.car_helper import convert_buckets_to_steer_angle
for i in range(10):
    idx = np.random.randint(50, 500)
    sample = X_train[idx:idx+TIME_STEPS*BATCH_SIZE, ...]
    sample = np.reshape(sample, newshape=[BATCH_SIZE, TIME_STEPS, HEIGHT, WIDTH, CHANNELS])
    start = time.clock()
    pred = rnn.model().predict(sample)[-1]
    b = convert_buckets_to_steer_angle(pred[0])
    end = time.clock()
    rnn.model().reset_states()
    print("Prediction in {} is  {} in {}".format(idx, b, (end - start)))

# plt.imshow(sample[-1][-1])
# plt.show()












