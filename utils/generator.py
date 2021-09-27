
import tensorflow as tf
import numpy as np
import os
try:
    import utils.config as config
except:
    import config as config


def get_xy(index, features_folder):
    x = np.zeros((len(index), config.TX, config.N_FREQ))
    y = np.zeros((len(index), config.TY, 1))
    for i, ind in enumerate(index):
        x[i] = np.load(os.path.join(
            features_folder, 'X_train', str(ind) + '.npy')).swapaxes(0, 1)
        y[i] = np.load(os.path.join(
            features_folder, 'Y_train', str(ind) + '.npy')).swapaxes(0, 1)
    return x, y


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, indices, batch_size=16, shuffle=True,
                 features_folder='Data'):
        self.batch_size = batch_size
        self.indices = indices
        self.shuffle = shuffle
        self.features_folder = features_folder
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        indexes = self.index[index *
                             self.batch_size: (index + 1) * self.batch_size]
        batch_indexes = [self.indices[k] for k in indexes]
        X, y = get_xy(batch_indexes, self.features_folder)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(self.index)


if __name__ == '__main__':
    indexes = [0, 1, 2, 3, 4, 6]
    gen = DataGenerator(indexes, batch_size=2)
    count = 0
    for x, y in gen:

        print(x.shape)
        print(y.shape)
        count += 1
        if count >= 3:
            break
