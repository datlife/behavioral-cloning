import pickle
from FLAGS import *
from utils.DataSet import DataSet
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as grid
plt.interactive(False)


data = DataSet(DRIVING_LOG, IMG_DIR, sequence=TIME_STEPS)

features, labels = data.build_train_data()
pickle.dump({'features': features, 'labels': labels}, open('train.p', 'wb'))

fig = plt.figure(figsize=(60, 60))
idx = 6
gs = grid.GridSpec(idx, idx)

# VISUALIZE FRAME + STEER VALUES for i in range(5):
for i in range(idx):
    for j in range(idx):
        r = np.random.choice(len(features))
        img = features[r]
        ax = fig.add_subplot(gs[i*idx + j])
        title = "Steering Angle: " + str(labels[r][0])
        ax.set_title(title)
        ax.axis('off')
        ax.imshow(img)

gs.tight_layout(fig)
plt.show()