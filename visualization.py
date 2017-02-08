import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.interactive(False)
from BatchGenerator import BatchGenerator
# ############ VISUALIZATION ############## #
# Plot the data  to analyze the
# differences in data distribution (Histogram)
##############################################
CSV_HEADER = ['center', 'left', 'right', 'steer_angle', 'throttle', 'speed']
df = pd.read_csv('./data/driving_log.csv', names=CSV_HEADER, index_col=False)
# plt.figure()
hist = df.steer_angle.value_counts()
# plt.bar(hist.index, hist.values, width=0.01)

y = hist.values
i = hist.index

from utils.car_helper import convert_steering_angle_to_buckets
y = [convert_steering_angle_to_buckets(i) for i in i]
plt.figure()
plt.bar(i, y, width=0.01)
plt.show()
# #
# data = pd.read_pickle('train2.p')
# y_train = data['labels']
# plt.figure()
#
# a = np.sort(np.reshape(y_train, (-1, 2))[0])
# fit = stats.norm.pdf(a, np.mean(a), np.std(a))
# plt.plot(a, fit, '-o')
# plt.hist(a, normed=True)      # use this to draw histogram of your data
# plt.show()


# https://github.com/mohankarthik/CarND-BehavioralCloning-P3
# https://github.com/ksakmann/CarND-BehavioralCloning
# https://yerevann.github.io/2016/06/26/combining-cnn-and-rnn-for-spoken-language-identification/
