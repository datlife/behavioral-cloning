import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
plt.interactive(False)

# ############ VISUALIZATION ############## #
# Plot the data  to analyze the
# differences in data distribution (Histogram)
##############################################

IMG_PATH = './data/IMG'
CSV_HEADERS = ['center', 'left', 'right', 'steer_angle', 'throttle', 'speed']

# Import data
df = pd.read_csv('./data/driving_log.csv', names=CSV_HEADERS, index_col=False)

# ############################# VISUALIZE FRAME + STEER VALUES ###############################
fig = plt.figure(figsize=(60, 60))
gs = grid.GridSpec(10, 5)

for i in range(10):
    panel = df.loc[df.steer_angle >= -1 + (i/10)]
    ans = panel.loc[panel.steer_angle < -1 + ((i+1)/10)]
    if len(ans):
        for j in range(5):
            img = None
            idx = np.random.choice(ans.index)
            img_path = os.path.join(ans.center[idx])
            img = cv2.imread(img_path)
            if img:
                ax = fig.add_subplot(gs[i*5 + j])
                ax.set_title(ans.steer_angle[idx])
                ax.imshow(img)
gs.tight_layout(fig)


# ############################# VISUAL STEERING ANGLES ######################################
hist = df.steer_angle.value_counts()

p = plt.figure()
left = p.add_subplot(1, 2, 1).bar(df.steer_angle.index, df.steer_angle.values, width=0.01)
left.set_title('Angle Values over Time')
right = p.add_subplot(122).bar(hist.index,  hist.values, width=0.01)
right.set_title('Steering Angle Histogram')
plt.show()

