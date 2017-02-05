from utils.DataSet import DataSet
import klepto
# http://stackoverflow.com/questions/17513036/pickle-dump-huge-file-without-memory-error

LOG_PATH = './data/driving_log.csv'
IMG_PATH = './data/IMG/'

# Import data
data = DataSet(log_path=LOG_PATH, img_dir_path=IMG_PATH)
X_train, y_train = data.get_train_data()

print('Saving Data....')
# I could not use pickle here for some reason
d = klepto.archives.dir_archive('train.p', cached=True, serialized=True)
d['features'] = X_train
d['labels'] = y_train
d.dump()
d.clear()
print('Finished')
