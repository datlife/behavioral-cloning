import pandas as pd
import matplotlib.pyplot as plt
plt.interactive(False)


KEEP_PROP = 0.2
LEARN_RATE = 0.001
EPOCHS = 10
IMG_PATH = './data/IMG'
CSV_HEADERS = ['center', 'left', 'right', 'steer_angle', 'throttle', 'speed']

# Import data
df = pd.read_csv('./data/driving_log.csv', names=CSV_HEADERS, index_col=False)
# Visualize Angles
p = plt.figure()
left = p.add_subplot(1, 2, 1).bar(df.steer_angle.index, df.steer_angle.values, width=0.01)
# Visualize histogram
hist = df.steer_angle.value_counts()
right = p.add_subplot(122).bar(hist.index,  hist.values, width=0.01)

plt.show()

# Data Pre-processing
# X_train = -1.0 + 2.0*X_train/255   # Shape (SequenceID, 160, 320, 3)

# Train


# Post-process angle
