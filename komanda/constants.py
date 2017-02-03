# CONSTANTS DEFINITIONS

# Frame Size
HEIGHT = 480
WIDTH = 640
CHANNELS = 3

# Hyper-Parameters
LEARN_RATE = 0.01
EPOCHS     = 5
KEEP_PROP  = 0.2

# Our training data follows the "interpolated.csv" format from Ross Wightman's scripts.
CSV_HEADER = "index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt".split(",")
OUTPUTS = CSV_HEADER[-6:-3]  # angle,torque,speed
OUTPUT_DIM = len(OUTPUTS)    # predict all features: steering angle, torque and vehicle speed


# RNNs are trained using (truncated) back-prop through time.

# 1. "SEQ_LEN"     : Length of an input (Back-prop through time)
# 2. "BATCH_SIZE"  : Number of sequence fragments used in one EPOCH
# 3. "LEFT_CONTEXT": Number of extra frames from the past that we append to the left of our input sequence.
#                    We need to do it because 3D convolution with "VALID" padding "eats" frames from the left,
#                    decreasing the "SEQ_LEN"
# Note : Actually we can use variable SEQ_LEN and BATCH_SIZE, they are set to constants only for simplicity).

# One should be careful here to maintain the model's causality.
SEQ_LEN = 10
BATCH_SIZE = 4
LEFT_CONTEXT = 5

# The parameters of the LSTM that keeps the model state.
RNN_SIZE = 32
RNN_PROJ = 32

