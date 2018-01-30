from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras import layers
from keras.layers import Activation
import keras.backend as K
import tensorflow as tf
from keras.applications.densenet import DenseNet

def conv_bn(filter,size,name,input):
    with tf.name_scope(name) as scope:
        conv = Conv2D(filter,size, padding='same', name=scope + 'conv',use_bias=False)(input)
        conv = BatchNormalization(name=scope + 'bn')(conv)
        conv = Activation('relu', name=scope + 'act')(conv)
    return conv

def dense_down(filter,size,name,input):
    with tf.name_scope(name) as scope:
        x = BatchNormalization(name=scope+'bn')(input)
        x = Activation('relu',name=scope+'act')(x)
        x = conv_bn(filter,size,scope+'c1',x)
        x = Conv2D(filter, size, padding='same',use_bias=False,name=scope + '2')(x)
        out = layers.concatenate([x,input],axis=-1)
        return out

def dense_up(filter,size,name,input1,input2):
    with tf.name_scope(name) as scope:
        up = BatchNormalization(name=scope+'bn')(input1)
        up = Activation('relu', name=scope + 'act')(up)
        up = UpSampling2D((2, 2), name=scope + 'up')(up)
        x = conv_bn(filter,size,scope+'con1',up)
        x = Conv2D(filter, size, padding='same', name=scope + 'conv2',activation='relu')(x)
        out = layers.concatenate([up,x,input2])
        return out

def inference(inputs):
    #conv0
    conv0 = Conv2D(filters=64,
                   kernel_size=(4,4),
                   strides=(4,4),
                   name='conv0',use_bias=False)(inputs)

    conv1 = dense_down(64,(3,3),'conv1',conv0) #dim=128
    pool1 = MaxPooling2D((2,2))(conv1)

    conv2 = dense_down(128,(3,3),'conv2',pool1) #dim=128+128
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = dense_down(256,(3,3),'conv3',pool2) #dim=256+256
    pool3 = MaxPooling2D((2, 2))(conv3)

    with tf.name_scope('conv5') as scope:
        center = BatchNormalization(name=scope+'bn1')(pool3)
        center = Activation('relu',name=scope+'act1')(center)
        center = conv_bn(512,(3,3),scope+'c1',center)
        center = Conv2D(256,(3,3),padding='same',name=scope+'c3')(center)

    conv7 = dense_up(256,(3,3),'conv7',center,conv3) #dim = 256+256+512 = 1024

    conv8 = dense_up(128, (3, 3), 'conv8',conv7,conv2) #dim = 1024+128+

    conv9 = dense_up(64, (3, 3), 'conv9', conv8, conv1)

    conv10 = conv_bn(64,(3,3),'conv10',conv9)

    conv11 = conv_bn(1, (1,1), 'conv11', conv10)

    return conv11

def main():
    input = Input(shape=(512,512,1))
    logit = inference(input)
    model = Model(input,logit)
    model.summary()
    sess = K.get_session()
    writer = tf.summary.FileWriter('logs/', sess.graph)

if __name__ == '__main__':
    main()

