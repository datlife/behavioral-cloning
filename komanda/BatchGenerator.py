import numpy as np
from .constants import *
# Input/output format
# Our data is presented as a long sequence of observations (several concatenated rosbags).
#
# We need to chunk it into a number of batches: for this, we will create BATCH_SIZE cursors.
# Let the starting points be uniformly spaced in our long sequence. We will advance them by SEQ_LEN at each step,
# creating a BATCH_SIZE x SEQ_LEN matrix of training examples. Boundary effects when one rosbag ends and
# the next starts are simply ignored.
# (Actually, LEFT_CONTEXT frames are also added to the left of the input sequence; see code below for details).

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

