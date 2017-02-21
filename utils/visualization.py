import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from model.BatchGenerator import BatchGenerator
plt.interactive(False)
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

#
# data = pd.read_pickle('./data/train.p')
# x_train = data['features']
# y_train = data['labels']
#
# x_train = x_train[0:2000]
# y_train = y_train[0:2000]
# x_train = np.reshape(x_train, newshape=[200, 10, 40, 160, 3])
# y_train = np.reshape(y_train, newshape=[200, 10, 2])
# gen = BatchGenerator(x_train, y_train, seq_len=50000, batch_size=5)
# batch_x, batch_y = gen.augment_data(x_train, y_train, scale=2)
#
# batch_y = np.reshape(batch_y, (-1, 2))
# y_train = np.reshape(y_train, (-1, 2))
#
# y_train = np.concatenate((y_train, y_train))
# print("Shape batch_y", np.shape(batch_y))
# plt.figure()
# a = plt.subplot(2, 1, 1)
# a.plot(y_train)
# b = plt.subplot(2, 1, 2)
# b.plot(batch_y)
# plt.show()
#
from utils.car_helper import convert_steering_angle_to_buckets, convert_buckets_to_steer_angle
#
a1 = [-1.0, -0.75, -0.5, -0.25,  -0.01, 0, 0.01, 0.25, 0.5, 0.75, 1.0]
a2 = [convert_steering_angle_to_buckets(i) for i in a1]
a3 = [convert_buckets_to_steer_angle(i) for i in a2]
print(a1)
print(a2)
print(a3)
y = [convert_steering_angle_to_buckets(i) for i in i]
plt.figure()
plt.bar(hist.index, y, width=0.01)

plt.figure()
a2 = [convert_steering_angle_to_buckets(i) for i in i]
item, count = np.unique(a2, return_counts=True)
freq = np.array((item, count)).T
plt.bar(item, count, alpha=0.6)
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
