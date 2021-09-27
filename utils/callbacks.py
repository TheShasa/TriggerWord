
import os
import datetime
import tensorflow as tf


def get_save_callback(folder='runs'):
    weights_path = folder+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    weights_path = weights_path+'/weights.h5'
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=weights_path,
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True)
    return save_callback


def get_tb_callback(folder='tb_logs'):
    tb_path = folder+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    tb_callback = tf.keras.callbacks.TensorBoard(tb_path, update_freq='epoch')
    return tb_callback
