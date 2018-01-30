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

def identity_block(inputs,filters,name):
    k1,k2,k3 = filters
    with tf.name_scope(name) as scope:
        x = Conv2D(k1, (1, 1), name=scope + 'v1')(inputs)
        x = BatchNormalization(name=scope + 'bn1')(x)
        x = Activation('relu', name=scope + 'act1')(x)

        x = Conv2D(k2,(3,3),name=scope+'v2',padding='same')(x)
        x = BatchNormalization(name=scope+'bn2')(x)
        x = Activation('relu',name=scope+'act2')(x)

        x = Conv2D(k3,(1,1),name=scope + 'v3')(x)
        x = BatchNormalization(name=scope+'bn3')(x)
        x = Activation('relu',name=scope+'act3')(x)

        out = layers.add([inputs,x])
        out = Activation('relu',name=scope+'act4')(out)
        return out

def conv_bn(filters,size,name,input):
    with tf.name_scope(name) as scope:
        x = Conv2D(filters, size,padding='same',name=scope+'conv1')(input)
        x = BatchNormalization(axis=3,name=scope+'bn1')(x)
        x = layers.Activation('relu',name=scope+'relu1')(x)

        x = Conv2D(filters, size, padding='same', name=scope + 'conv2')(x)
        x = BatchNormalization(axis=3, name=scope + 'bn2')(x)
        x = layers.Activation('relu', name=scope + 'relu2')(x)

        x = layers.concatenate([input,x],name=scope+'append')

        return x

def conv(filters,size,name,input):
    with tf.name_scope(name) as scope:
        x = Conv2D(filters, size,padding='same',name=scope+'conv1')(input)
        #x = BatchNormalization(axis=3,name=scope+'bn1')(x)
        x = layers.Activation('relu',name=scope+'relu1')(x)

        x = Conv2D(filters, size, padding='same', name=scope + 'conv2')(x)
        #x = BatchNormalization(axis=3, name=scope + 'bn2')(x)
        x = layers.Activation('relu', name=scope + 'relu2')(x)

        x = layers.concatenate([input,x],name=scope+'append')

        return x

def inference(inputs):

    conv0 = Conv2D(64,(3,3),padding='same',name='conv0',activation='relu')(inputs)

    conv1 = conv(64,(3,3),'conv1',conv0)
    pool1 = MaxPooling2D((2,2))(conv1) #dim=128

    conv2 = conv(128,(3,3),'conv2',pool1)
    pool2 = MaxPooling2D((2,2))(conv2) #dim=256

    conv3 = conv(256,(3,3),'conv3',pool2)
    pool3 = MaxPooling2D((2,2))(conv3) #dim=512

    center = Conv2D(512,(3,3), padding='same',activation='relu',name='center1')(pool3)
    center = Conv2D(512, (3, 3), padding='same', activation='relu', name='center2')(center)

    conv7 = UpSampling2D((2,2))(center)
    conv7 = layers.concatenate([conv7,conv3])
    conv7 = conv(256,(3,3),'conv7',conv7) #dim=512

    conv8 = UpSampling2D((2, 2))(conv7)
    conv8 = layers.concatenate([conv8, conv2])
    conv8 = conv(128,(3,3),'conv8', conv8)  # dim=128

    conv9 = UpSampling2D((2, 2))(conv8)
    conv9 = layers.concatenate([conv9, conv1])
    conv9 = conv(64,(3,3),'conv9', conv9)  # dim=64

    with tf.name_scope('conv10') as scope:
        x = Conv2D(64,(3,3),padding='same',name=scope+'conv1')(conv9)
        x = BatchNormalization(axis=3, name=scope + 'bn1')(x)
        x = layers.Activation('relu', name=scope + 'act1')(x)

        x = Conv2D(1,(1,1),name=scope+'conv2')(x)
        x = BatchNormalization(axis=3, name=scope + 'bn2')(x)
        x = layers.Activation('sigmoid', name=scope + 'act2')(x)

    return x



def main():
    input = Input(shape=(256,256,1))
    logit = inference(input)
    model = Model(input,logit)
    model.summary()
    sess = K.get_session()
    writer = tf.summary.FileWriter('logs/', sess.graph)

if __name__ == '__main__':
    main()

