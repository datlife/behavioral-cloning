# define some constants

# RNNs are typically trained using (truncated) backprop through time. SEQ_LEN here is the length of BPTT.
# Batch size specifies the number of sequence fragments used in a sigle optimization step.
# (Actually we can use variable SEQ_LEN and BATCH_SIZE, they are set to constants only for simplicity).
# LEFT_CONTEXT is the number of extra frames from the past that we append to the left of our input sequence.
# We need to do it because 3D convolution with "VALID" padding
# "eats" frames from the left, decreasing the sequence length.

# One should be careful here to maintain the model's causality.
SEQ_LEN = 10
BATCH_SIZE = 4
LEFT_CONTEXT = 5

# These are the input image parameters.
HEIGHT = 480
WIDTH = 640
CHANNELS = 3    # RGB

# The parameters of the LSTM that keeps the model state.
RNN_SIZE = 32
RNN_PROJ = 32

# Our training data follows the "interpolated.csv" format from Ross Wightman's scripts.
CSV_HEADER = "index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt".split(",")
OUTPUTS = CSV_HEADER[-6:-3]  # angle,torque,speed
OUTPUT_DIM = len(OUTPUTS)    # predict all features: steering angle, torque and vehicle speed
