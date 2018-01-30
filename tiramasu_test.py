from keras.layers import Conv2D,Conv2DTranspose
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import Dropout,concatenate
from keras.regularizers import l2
from keras.layers import Activation
import keras.backend as K
import tensorflow as tf

def denseBlock(t, nb_layers,name):
    with tf.name_scope(name) as scope:
        for i in range(nb_layers):
            tmp = t
            t = BatchNormalization(axis=3,
                                   gamma_regularizer=l2(0.0001),
                                   beta_regularizer=l2(0.0001),
                                   name=scope+'bn'+str(i))(t)
            t = Activation('relu',name=scope+'act'+str(i))(t)
        # the filter of conv means grouth rate
            t = Conv2D(16, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform',
                       data_format='channels_last',name=scope+'c'+str(i))(t)
            t = Dropout(0.,name=scope+'d'+str(i))(t)
            t = concatenate([t, tmp])
    return t

def transitionDown(t, nb_features,name):
    with tf.name_scope(name) as scope:
        t = BatchNormalization(axis=3,
                            gamma_regularizer=l2(0.0001),
                            beta_regularizer=l2(0.0001),
                            name = scope + 'bn')(t)
        t = Activation('relu',name = scope + 'act')(t)
        t = Conv2D(nb_features, kernel_size=(1, 1), padding='same', kernel_initializer='he_uniform',
                   data_format='channels_last',name = scope + 'c')(t)
        t = Dropout(0.,name = scope + 'd')(t)
        t = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same',
                         data_format='channels_last',name = scope + 'p')(t)
    return t

def transitionUp(t, nb_features,name):
    with tf.name_scope(name) as scope:
        t = Conv2DTranspose(nb_features,
                            strides=2,
                            kernel_size=(3, 3),
                            padding='same',
                            name=scope+'c')(t)  # transition Up
    return t

'''
    the output of dense,if input dim=48 each convdim=16
    the out_put = 48 + 16*4 = 112 
    thats papers describe:
    layer1:48
    layer2:layer1+16*5
    layer3:layer2+16*5
    etc 
    '''

def inference(inputs):
    n_pool = 3
    growth_rate = 16
    layer_per_block = [5,5,5,15,5,5,5]
    nb_features = 48
    t = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer='he_uniform',name='conv0')(inputs)
    #down sampleing
    skip_connections = []  # save the shape of Transpose_conv
    for i in range(n_pool):
        t = denseBlock(t, layer_per_block[i],'DB'+str(i))
        skip_connections.append(t) #save the layer to connect
        nb_features += growth_rate * layer_per_block[i] #clever
        t = transitionDown(t,nb_features,'TD'+str(i))
    # center
    t = denseBlock(t,layer_per_block[n_pool],'center') # bottle neck
    #up sampling
    skip_connections = list(reversed(skip_connections))
    for i in range(n_pool):
        iter = n_pool+i+1
        keep_nb_features = growth_rate * layer_per_block[n_pool + i]
        t = transitionUp(t,keep_nb_features,'TP'+str(iter))
        t = concatenate([t, skip_connections[i]])
        t = denseBlock(t, layer_per_block[iter],name='DB'+str(iter))

    with tf.name_scope('output') as scope:
        t = BatchNormalization(axis=3,
                            gamma_regularizer=l2(0.0001),
                            beta_regularizer=l2(0.0001),
                            name = scope + 'bn')(t)
        t = Activation('relu', name=scope + 'act')(t)
        output = Conv2D(1,(1,1), padding='same', activation='sigmoid',
                        kernel_initializer='he_uniform',name=scope+'c')(t)
    return output


def main():
    input = Input(shape=(128,128,1))
    logit = inference(input)
    model = Model(input,logit)
    model.summary()
    sess = K.get_session()
    writer = tf.summary.FileWriter('logs/',sess.graph)

if __name__ == '__main__':
    main()