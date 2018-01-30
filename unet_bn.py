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

def conv_bn(filters,size,name,input):
    with tf.name_scope(name) as scope:
        x = Conv2D(filters, size,padding='same',name=scope+'conv')(input)
        x = BatchNormalization(axis=3,name=scope+'bn')(x)
        x = layers.Activation('relu',name=scope+'relu')(x)
        return x

def conv(filters,size,name,input):
    with tf.name_scope(name) as scope:
        x = Conv2D(filters, size,padding='same',name=scope+'conv')(input)
        x = layers.Activation('relu',name=scope+'relu')(x)
        return x


def inference(inputs):
    conv1 = conv(64,(3,3),'conv1_1',inputs)
    conv1 = conv_bn(64,(3,3),'conv1_2',conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    #conv2 = conv(64,(3,3),'conv2_1',pool1)
    #conv2 = conv(64,(3,3),'conv2_2',conv2)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv(128,(3,3),'conv3_1',pool1)
    conv3 = conv_bn(128,(3,3),'conv3_2',conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv(256,(3,3),'conv4_1',pool3)
    conv4 = conv_bn(256,(3,3),'conv4_2',conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    center = conv(512,(3,3),'center1',pool4)
    center = conv_bn(512,(3,3),'center2',center)

    up6 = conv(256,(2,2),'conv6_1',UpSampling2D(size=(2, 2))(center))
    # merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
    merge6 = layers.concatenate([conv4, up6], axis=-1)
    conv6 = conv(256,(3,3),'conv6_2',merge6)
    conv6 = conv_bn(256,(3,3),'conv6_2',conv6)

    up7 = conv(128, (2, 2), 'conv7_1', UpSampling2D(size=(2, 2))(conv6))
    # merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
    merge7 = layers.concatenate([conv3, up7], axis=-1)
    conv7 = conv(128, (3, 3), 'conv7_2', merge7)
    conv7 = conv_bn(128, (3, 3), 'conv7_2', conv7)

    #up8 = conv(64, (2,2), 'conv8_1', UpSampling2D(size=(2, 2))(conv7))
    #merge8 = layers.concatenate([conv2, up8], axis=-1)
    #conv8 = conv(64, (3, 3), 'conv8_2', merge8)
    #conv8 = conv(64, (3, 3), 'conv8_2', conv8)

    up9 = conv(64,(2,2),'conv9_1',UpSampling2D(size=(2, 2))(conv7))
    # merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
    #merge9 = layers.concatenate([conv1, up9], axis=-1)
    merge9 = layers.concatenate([conv1,up9], axis=-1)
    conv9 = conv(64,(3,3),'conv9_2',merge9)
    conv9 = conv_bn(64,(3,3),'conv9_3',conv9)

    with tf.name_scope('conv10') as scope:
        conv10 = layers.Conv2D(filters=1,kernel_size=(1,1),name=scope+'conv')(conv9)
        conv10 = layers.BatchNormalization(axis=3,name=scope+'bn')(conv10)
        conv10 = layers.Activation('sigmoid')(conv10)

    return conv10


def main():
    input = Input(shape=(128,128,1))
    logit = inference(input)
    model = Model(input,logit)
    model.summary()
    sess = K.get_session()
    writer = tf.summary.FileWriter('logs/',sess.graph)

if __name__ == '__main__':
    main()


