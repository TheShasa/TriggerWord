
from utils.model import create_model
from utils.generator import DataGenerator
import utils.config as config
from utils.callbacks import get_save_callback, get_tb_callback
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
tf.compat.v1.disable_v2_behavior()  # model trained in tf1

model = create_model(input_shape=(config.TX, config.N_FREQ))
model.summary()

opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


path_to_x = 'Data/X_train/'
path_to_y = 'Data/Y_train/'


indexes = np.arange(config.TRAIN_SIZE)
np.random.shuffle(indexes)

train_index = indexes[config.TRAIN_SIZE*config.SPLIT_VAL_PERCENT//100:]
val_index = indexes[:config.TRAIN_SIZE*config.SPLIT_VAL_PERCENT//100]


train_generator = DataGenerator(train_index, config.BATCH_SIZE)
val_generator = DataGenerator(val_index, batch_size=config.BATCH_SIZE,
                              shuffle=False)
for x, y in train_generator:
    print(x.shape, y.shape)
    break
for x, y in val_generator:
    print(x.shape, y.shape)
    break

save_callback = get_save_callback()

tb_callback = get_tb_callback()
model.fit(train_generator, validation_data=val_generator,
          epochs=config.NUM_EPOCHS, callbacks=[save_callback, tb_callback])
