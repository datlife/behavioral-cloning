import cv2
import os
import pandas as pd
import numpy as np
from utils.image_processor import random_transform
from scipy.misc import imresize
# # git+https://github.com/uqfoundation/pathos.git@master
# http://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma


class DataSet(object):
    ######################################
    # Process data from CSV(s) into
    # Training, Validation data set
    # =============================
    #
    ######################################
    CSV_HEADER = ['center', 'left', 'right', 'steer_angle', 'throttle', 'speed']

    def __init__(self, log_path, img_dir_path, sequence=10):
        self.df = pd.read_csv(log_path, names=self.CSV_HEADER, index_col=False)
        self.img_path = img_dir_path
        self.sequence_len = sequence
        self.X_train = []
        self.y_train = []

    def build_train_data(self):
        """
        :return:
        """
        print('Loading training data...')
        for i in range((len(self.df))):
            panorama_img = None
            # Create panorama frame
            # Merge 3 images into panorama image [160, 320*3, 3]
            # for image in ['right', 'center', 'left']:
            for image in ['center']:
                img_file = self.df.loc[i][image].rsplit('/')[-1]  # Extract image file only
                img = cv2.imread(os.path.join(self.img_path, img_file))
                img = imresize(img, 0.5)
                img = img[30:80, :, :]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if panorama_img is None:
                    panorama_img = img
                else:
                    panorama_img = np.hstack((img, panorama_img)).astype('uint8')

            self.X_train.append(panorama_img)
            # steer_angle - throttle
            label = (self.df.loc[i]['steer_angle'], self.df.loc[i]['throttle'])
            self.y_train.append(np.array(label, dtype=float))

        print("Data loaded.")
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        print("Input shape: ", np.shape(self.X_train))
        print("Label shape: ", np.shape(self.y_train))
        return self.X_train, self.y_train


    def get_data(self):
        return self.X_train, self.y_train
# LOG_PATH = './data/driving_log.csv'
# IMG_PATH = './data/IMG/'
#
# import numpy as np
#
# # Import data
# data = DataSet(log_path=LOG_PATH, img_dir_path=IMG_PATH, sequence=10)
# X_train, y_train = data.build_train_data()
# plt.imshow(X_train[100])
# plt.show()
# # Save data
# print('Saving Data....')
# pickle.dump({'features': X_train, 'labels': y_train}, open('train.p', 'wb'))
# print('Finished')
