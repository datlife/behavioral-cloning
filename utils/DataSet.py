import cv2
import os
import pandas as pd
import numpy as np
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

    def __init__(self, log_path, img_dir_path):
        self.df = pd.read_csv(log_path, names=self.CSV_HEADER, index_col=False)
        self.X_train = []
        self.y_train = []
        self.img_path = img_dir_path

    def get_train_data(self):
        """
        :return:
        """
        print('Loading training data...')
        for i in range(100):
        # for i in range((len(self.df))):
            panorama_img = None
            # Merge 3 images into panorama image [160, 320*3, 3]
            for image in ['right', 'center', 'left']:
                img_file = self.df.loc[i][image].rsplit('/')[-1]  # Extract image file only
                img = cv2.imread(os.path.join(self.img_path, img_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if panorama_img is None:
                    panorama_img = img
                else:
                    panorama_img = np.hstack((img, panorama_img)).astype(int)

            # Training image [160, 960, 3]
            self.X_train.append(panorama_img)
            # steer_angle - throttle
            self.y_train.append(np.array(self.df.loc[i][['steer_angle', 'throttle']], dtype=float))
        print('Finished')

        return np.array(self.X_train), np.array(self.y_train)



