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
from keras.regularizers import l2
from keras.applications.resnet50 import ResNet50

def res_block(filters,name,inputs):
    with tf.name_scope(name) as scope:
        x = Conv2D(filters,(3,3),name=scope+'c1',kernel_initializer='he_uniform',padding='same')(inputs)
        x = BatchNormalization(axis=3,
                               gamma_regularizer=l2(0.0001),
                               beta_regularizer=l2(0.0001),
                               name=scope + 'b1')(x)
        x = Activation('relu',name=scope+'a1')(x)
        x = Conv2D(filters,(3,3),name=scope + 'c2',kernel_initializer='he_uniform',padding='same')(x)
        x = BatchNormalization(axis=3,
                               gamma_regularizer=l2(0.0001),
                               beta_regularizer=l2(0.0001),
                               name=scope + 'b2')(x)
        out = layers.add([inputs,x])
        out = Activation('relu',name=scope+'act3')(out)
        return out

def conv_bn(filters,size,name,inputs):
    with tf.name_scope(name) as scope:
        x = Conv2D(filters, size,padding='same',name=scope+'c')(inputs)
        x = BatchNormalization(axis=3,
                               gamma_regularizer=l2(0.0001),
                               beta_regularizer=l2(0.0001),
                               name=scope + 'b')(x)
        x = layers.Activation('relu',name=scope+'a')(x)
        return x

def transitionDown(nb_features,name,inputs):
    with tf.name_scope(name) as scope:
        t = Conv2D(nb_features, kernel_size=(1,1), padding='same', kernel_initializer='he_uniform',
                   data_format='channels_last',name = scope + 'c')(inputs)
        t = BatchNormalization(axis=3,
                               gamma_regularizer=l2(0.0001),
                               beta_regularizer=l2(0.0001),
                               name=scope + 'bn')(t)
        t = Activation('relu', name=scope + 'act')(t)
        t = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same',
                         data_format='channels_last',name = scope + 'p')(t)
    return t

def transitionUp(nb_features,name,inputs):
    with tf.name_scope(name) as scope:
        t = UpSampling2D((2,2),name=scope+'up')(inputs)
        t = Conv2D(nb_features, kernel_size=(1,1), padding='same', kernel_initializer='he_uniform',
                   data_format='channels_last',name = scope + 'c')(t)
        t = BatchNormalization(axis=3,
                               gamma_regularizer=l2(0.0001),
                               beta_regularizer=l2(0.0001),
                               name=scope + 'bn')(t)
        t = Activation('relu', name=scope + 'act')(t)
    return t

def inference(inputs):
    #conv0
    t = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
               kernel_initializer='he_uniform', name='conv0')(inputs)
    t = BatchNormalization(axis=-1, name='bn_conv1')(t)
    t = Activation('relu')(t)
    t = MaxPooling2D((2, 2))(t)

    conv1 = res_block(64,'res1',t)
    pool1 = transitionDown(128,'pool1',conv1)

    conv2 = res_block(128,'res2',pool1)
    pool2 = transitionDown(256,'pool2',conv2)

    conv3 = res_block(256, 'res3', pool2)
    pool3 = transitionDown(512, 'pool2', conv3)

    center = res_block(512,'center',pool3)

    up6 = transitionUp(256,'up6',center)
    c6 = layers.concatenate([up6,conv3])
    c6 = conv_bn(256,(3,3),'c6',c6)
    res6 = res_block(256,'res6',c6)

    up7 = transitionUp(128, 'up7', res6)
    c7 = layers.concatenate([up7, conv2])
    c7 = conv_bn(128, (3, 3), 'c7', c7)
    res7 = res_block(128, 'r7', c7)

    up8 = transitionUp(64, 'up8', res7)
    c8 = layers.concatenate([up8, conv1])
    c8 = conv_bn(64, (3, 3), 'c8', c8)
    res8 = res_block(64, 'res8', c8)

    with tf.name_scope('output') as scope:
        x = Conv2D(1,(3,3),padding='same',name=scope+'c')(res8)
        x = BatchNormalization(axis=3,
                               gamma_regularizer=l2(0.0001),
                               beta_regularizer=l2(0.0001),
                               name=scope + 'b')(x)
        output = layers.Activation('sigmoid',name=scope+'a')(x)
    return output

def main():
    input = Input(shape=(512,512,1))
    logit = inference(input)
    model = Model(input,logit)
    model.summary()
    sess = K.get_session()
    writer = tf.summary.FileWriter('logs/', sess.graph)

if __name__ == '__main__':
    main()

