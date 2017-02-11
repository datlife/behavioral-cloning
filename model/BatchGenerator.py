# This code is based on the winner of Udacity Challenge 2 Winner
# Input/output format
# Our data is presented as a long sequence of observations (several concatenated ros-bag).
#
# 1. "SEQ_LEN"     : Length of an input (Back-prop through time)
# 2. "BATCH_SIZE"  : Number of sequence fragments used in one EPOCH
# 3. "LEFT_CONTEXT": Number of extra frames from the past that we append to the left of our input sequence.
#                    We need to do it because 3D convolution with "VALID" padding "eats" frames from the left,
#                    decreasing the "SEQ_LEN"
# Note : Actually we can use variable SEQ_LEN and BATCH_SIZE, they are set to constants only for simplicity).
import numpy as np
LEFT_CONTEXT = 5
from FLAGS import *
import cv2
from utils.image_processor import random_transform
from utils.car_helper import convert_steering_angle_to_buckets


class BatchGenerator(object):
    def __init__(self, sequence, labels, seq_len, batch_size):
        self.sequence = sequence
        self.labels = labels
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.cursor = 0

    def next(self, scale=1):
        '''
        Divide sequence of images into a block size (BATCH, TIME_STEPS, HEIGHT, WEIGHT, CHANNELS)
        :param scale:
        :return:
        '''
        while True:
            x_train = None
            y_train = None
            for i in range(self.batch_size):
                frame_block = self.sequence[(i+self.cursor):(i + self.cursor + self.seq_len)]
                labels = self.labels[(i+self.cursor):(i + self.seq_len + self.cursor)]
                if x_train is None:
                    x_train = frame_block
                    y_train = labels
                else:
                    x_train = np.concatenate((x_train, frame_block))   # batch_size x seq_len
                    y_train = np.concatenate((y_train, labels))        # batch_size x seq_len x OUTPUT_DIM

            # Update batch cursor
            self.cursor += self.batch_size*self.seq_len

            # Convert into blocks of sequence
            x_train = np.reshape(x_train, newshape=[BATCH_SIZE, TIME_STEPS, HEIGHT, WIDTH, CHANNELS])
            y_train = np.reshape(y_train, newshape=[BATCH_SIZE, TIME_STEPS, OUTPUT_DIM])

            # To make sure that end of block is matched the batch size steps
            steps = round(len(x_train)/BATCH_SIZE)
            x_train = x_train[0: steps*BATCH_SIZE]
            y_train = y_train[0: steps*BATCH_SIZE]

            # Data augmentation
            x_train, y_train = self.augment_data(x_train, y_train, scale=scale)

            if OUTPUT_DIM > 1:
                y_train = np.mean(y_train, axis=1)

            return x_train, y_train

    def augment_data(self, X_train, y_train, scale):
        """
        Perform data augmentation in block
        SHOULD BE IN CALL BACK OF fit_generator REAL TIME
        :param scale:
        :return:
        """
        for i in range(scale-1):
            for b in range(len(X_train)):
                block = X_train[b]
                augmented_block = []
                augmented_label = y_train[b]

                # For each frame in block:
                #   1-st augmentation: flip it and build a new frame block
                #   2-st augmentation: random transform
                for idx, frame in enumerate(block):
                    if i == 0:
                        new_frame = cv2.flip(frame, 1)
                        augmented_label[idx][0] *= -1.0
                    if i > 0:
                        new_frame = random_transform(new_frame)
                    augmented_block.append(new_frame)

                # Add new frame block to training data
                X_train = np.concatenate((X_train, np.expand_dims(np.array(augmented_block), axis=0)))
                y_train = np.concatenate((y_train, np.expand_dims(augmented_label, axis=0)))

        return X_train, y_train

    def get_seq_size(self):
        return len(self.sequence) - BATCH_SIZE*TIME_STEPS

    def cursor(self):
        return self.cursor

    def reset_cursor(self):
        self.cursor = 0
