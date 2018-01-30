from tiramasu56_nodropout import inference
import cv2
from keras import Model
from keras.layers import Input
from scipy import misc
import numpy as np

def read_image(path):
    result = misc.imread(path)
    #result = misc.imresize(result,[512,512])
    result = np.expand_dims(result,axis=-1)
    return result

def main():
    img_path = 'data/img/volume-76.nii/104.jpg'
    seg_path = 'data/seg/segmentation-76.nii/104.jpg'
    img = read_image(img_path)
    seg = misc.imread(seg_path)
    img_t = np.expand_dims(img,axis=0)

    input = Input(shape=[512,512,1])
    logit = inference(input)
    model = Model(input,logit)
    model.load_weights('weights/100epoch.h5')
    result = model.predict(img_t)

    result = np.squeeze(result,axis=0)
    result = np.squeeze(result,axis=-1)
    result = result*255.
    result = result.astype(np.uint8)
    result = misc.imresize(result,[512,512])
    print(np.max(result))

    #show the image
    img = misc.imread(img_path)

    cv2.imshow('img',img)
    cv2.imshow('predict',result)
    cv2.imshow('true',seg)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()


