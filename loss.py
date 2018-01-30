from keras import layers
from keras import losses
from keras import metrics
import keras.backend as K

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = layers.Flatten()(y_true)
    y_pred_f = layers.Flatten()(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    result = 1.-dice_coef(y_true, y_pred)
    return result

def map_accuracy(y_true, y_pred):
    y_true_f = layers.Flatten()(y_true)
    y_pred_f = layers.Flatten()(y_pred)
    acc = metrics.binary_accuracy(y_true_f,y_pred_f)
    return acc