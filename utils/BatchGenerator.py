from constants import *
import numpy as np
# Input/output format
# Our data is presented as a long sequence of observations (several concatenated ros-bag).
#
# 1. "SEQ_LEN"     : Length of an input (Back-prop through time)
# 2. "BATCH_SIZE"  : Number of sequence fragments used in one EPOCH
# 3. "LEFT_CONTEXT": Number of extra frames from the past that we append to the left of our input sequence.
#                    We need to do it because 3D convolution with "VALID" padding "eats" frames from the left,
#                    decreasing the "SEQ_LEN"
# Note : Actually we can use variable SEQ_LEN and BATCH_SIZE, they are set to constants only for simplicity).


class BatchGenerator(object):
    def __init__(self, sequence, seq_len, batch_size):
        self.sequence = sequence
        self.seq_len = seq_len
        self.batch_size = batch_size
        chunk_size = 1 + (len(sequence) - 1) / batch_size
        self.indices = [(i * chunk_size) % len(sequence) for i in range(batch_size)]

    def next(self):
        while True:
            output = []
            for i in range(self.batch_size):
                idx = self.indices[i]
                left_pad = self.sequence[idx - LEFT_CONTEXT:idx]
                if len(left_pad) < LEFT_CONTEXT:
                    left_pad = [self.sequence[0]] * (LEFT_CONTEXT - len(left_pad)) + left_pad
                assert len(left_pad) == LEFT_CONTEXT
                leftover = len(self.sequence) - idx
                if leftover >= self.seq_len:
                    result = self.sequence[idx:idx + self.seq_len]
                else:
                    result = self.sequence[idx:] + self.sequence[:self.seq_len - leftover]
                assert len(result) == self.seq_len
                self.indices[i] = (idx + self.seq_len) % len(self.sequence)
                images, targets = zip(*result)
                images_left_pad, _ = zip(*left_pad)
                output.append((np.stack(images_left_pad + images), np.stack(targets)))
            output = zip(*output)
            output[0] = np.stack(output[0])  # batch_size x (LEFT_CONTEXT + seq_len)
            output[1] = np.stack(output[1])  # batch_size x seq_len x OUTPUT_DIM
            return output
