# One should be careful here to maintain the model's causality.
SEQ_LEN = 10
BATCH_SIZE = 4
LEFT_CONTEXT = 5

# These are the input image parameters.
HEIGHT = 480
WIDTH = 640
CHANNELS = 3
video_frame = [WIDTH, HEIGHT, CHANNELS]

# The parameters of the LSTM that keeps the model state.
RNN_SIZE = 32
RNN_PROJ = 32

# Our training data follows the "interpolated.csv" format from Ross Wightman's scripts.
CSV_HEADER = "index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt".split(",")
OUTPUTS = CSV_HEADER[-5:-4]  # angle,torque
OUTPUT_DIM = len(OUTPUTS)    # predict all features: steering angle, torque and vehicle speed
KEEP_PROP = 0.2