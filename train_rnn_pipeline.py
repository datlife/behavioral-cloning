import pickle
from FLAGS import *                                   # Stores parameters and hy
from keras.optimizers import Adam
from model.BatchGenerator import BatchGenerator       # Generate batch file to feed into my RNN
from model.DatNet import DatNet, mse_steer_angle      # Compute the loss of steering angle
from keras.models import model_from_json

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
#
with open('model.json', 'r') as json_file:
    json_model = json_file.read()
    model.RNN = model_from_json(json_model)
print("Load model successfully")
model.RNN.load_weights('model.h5')

# # Compile
# # TODO:
# # Find better loss function
model.RNN.compile(optimizer=Adam(lr=0.001), loss=[mse_steer_angle], metrics=['mse'])
model.train_rnn(batch_generator=batch_gen, epochs=2, augmentation_scale=1)
model.RNN.summary()

# Save model
json_string = model.RNN.to_json()
with open('model.json', 'w') as outfile:
    outfile.write(json_string)
model.RNN.save_weights('model.h5')
print('Model saved')

#