# Image Size
HEIGHT = 50
WIDTH = 160
CHANNELS = 3
# output dimension from RNN
OUTPUT_DIM = 2
# RNN SIZE
HIDDEN_UNITS = 32

# HYPER-PARAMETER
LEARN_RATE = 0.0001
KEEP_PROP = 0.3
EPOCHS = 10
BATCH_SIZE = 2    # Be careful, stateful RNN requires to state batch size ahead
TIME_STEPS = 5   # For RNN
INIT = 'he_uniform'

# INPUT
DRIVING_LOG = './data/driving_log.csv'
IMG_DIR = './data/IMG'