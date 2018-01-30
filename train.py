from __future__ import print_function
import tensorflow as tf
from keras import callbacks
from keras.models import Model
from read_data import file_to_tensor,get_path
from tiramasu56 import inference
import keras.optimizers as opt
from keras import metrics
from keras import layers,losses
import numpy as np
from loss import dice_coef_loss,map_accuracy

flags = tf.app.flags
flags.DEFINE_integer('batch_size',4,'batch_size')
flags.DEFINE_string('weight','false','weight')
flags.DEFINE_integer('epoch',50,'epoch')
flags.DEFINE_float('lr',1e-5,'lr')
FLAGS = flags.FLAGS

#pay attention to main(_)
def main(_):
    batch_size = FLAGS.batch_size
    epoch = FLAGS.epoch
    weight = FLAGS.weight
    lr = FLAGS.lr

    path,_ = get_path()
    steps_per_epoch = int(np.ceil(len(path)/(float(batch_size))))
    img,seg = file_to_tensor(batch_size=batch_size,
                             epoch=epoch)
    #???? r u kidding me?
    inputs = layers.Input(tensor=img)
    logit = inference(inputs)
    model = Model(inputs,logit)
    ckpt = callbacks.ModelCheckpoint(filepath='weights/last_weight.h5',
                                     period=1,
                                     save_weights_only=True)
    tb = callbacks.TensorBoard(log_dir='logs/',write_graph=True)

    model.compile(optimizer=opt.rmsprop(),
                        loss=dice_coef_loss,
                        metrics=[map_accuracy],
                        target_tensors=[seg])
    if weight!='false':
        model.load_weights(weight)

    model.summary()
    model.fit(epochs=epoch,
              steps_per_epoch=steps_per_epoch,
              verbose=1,
              callbacks=[ckpt])

if __name__ == '__main__':
    tf.app.run()
