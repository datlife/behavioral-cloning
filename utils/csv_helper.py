import csv
import numpy as np
# One should be careful here to maintain the model's causality.
SEQ_LEN = 10
BATCH_SIZE = 128
LEFT_CONTEXT = 5
CSV_HEADER2 = "center_cam, left_cam, right_cam, angle, throttle, speed"
OUTPUTS = CSV_HEADER2[-2:-3]  # angle,torque
OUTPUT_DIM = len(OUTPUTS)    # predict all features: steering angle, torque and vehicle speed


def process_csv(filename, image_path, val=5):
    sum_f = np.float128([0.0] * OUTPUT_DIM)
    sum_sq_f = np.float128([0.0] * OUTPUT_DIM)
    lines = read_csv(filename)
    # leave val% for validation
    train_seq = []
    valid_seq = []
    cnt = 0
    for ln in lines:
        if cnt < SEQ_LEN * BATCH_SIZE * (100 - val):
            train_seq.append(ln)
            sum_f += ln[1]
            sum_sq_f += ln[1] * ln[1]
        else:
            valid_seq.append(ln)
        cnt += 1
        cnt %= SEQ_LEN * BATCH_SIZE * 100
    mean = sum_f / len(train_seq)
    var = sum_sq_f / len(train_seq) - mean * mean
    std = np.sqrt(var)
    print(len(train_seq), len(valid_seq))
    print(mean, std)  # we will need these statistics to normalize the outputs (and ground truth inputs)
    return (train_seq, valid_seq), (mean, std)