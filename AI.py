from tiramasu56_nodropout import inference
import cv2
from keras import Model
from keras.layers import Input
from scipy import misc
import numpy as np
from skimage import measure
from skimage.morphology import rectangle
from read_data import get_path
import os
import random

class bot:
    def __init__(self):
        input = Input(shape=[512, 512, 1])
        logit = inference(input)
        self.model = Model(input, logit)
        self.model.load_weights('weights/100epoch.h5')
        print('init finish')

    def read_image(self,path):
        result = misc.imread(path)
        # result = misc.imresize(result,[512,512])
        result = np.expand_dims(result, axis=-1)
        return result

    def predict(self,path):
        try:
            img_path = path
            img = self.read_image(img_path)
            org = img.copy()
            img_t = np.expand_dims(img, axis=0)
            mask = self.model.predict(img_t)
            mask = np.squeeze(mask, axis=0)
            mask = np.squeeze(mask, axis=-1)
            mask = misc.imresize(mask, [512, 512])
            #union
            print(mask.dtype)
            mask = cv2.erode(mask,rectangle(3,3))
            edge = cv2.Canny(mask,30,100)
            edge = cv2.dilate(edge,rectangle(3,3))
            org = cv2.cvtColor(org,cv2.COLOR_GRAY2BGR)
            org[edge!=0] = [0,127,255]
            cv2.imwrite('temp/mask.jpg',mask)
            cv2.imwrite('temp/edge.jpg',edge)
            cv2.imwrite('temp/2.jpg',org)
            result = 1
            print('inference sucess!')
        except:
            result = 0
            print('inference failed!')
        return result

def main():
    ai = bot()
    path = 'data/img/volume-90.nii/374.jpg'
    ai.predict(path)


def main2():
    img,_ = get_path()
    ai = bot()
    evals = random.sample(img,20)
    for i in range(len(evals)):
        print(i)
        img = ai.read_image(evals[i])
        cv2.imwrite('result/{}_img.jpg'.format(i),img)
        org = img.copy()
        img_t = np.expand_dims(img, axis=0)
        mask = ai.model.predict(img_t)
        mask = np.squeeze(mask, axis=0)
        mask = np.squeeze(mask, axis=-1)
        mask = misc.imresize(mask, [512, 512])
        # union
        #print(mask.dtype)
        mask = cv2.erode(mask, rectangle(3, 3))
        edge = cv2.Canny(mask, 30, 100)
        edge = cv2.dilate(edge, rectangle(3, 3))
        org = cv2.cvtColor(org, cv2.COLOR_GRAY2BGR)
        org[edge != 0] = [0, 127, 255]
        cv2.imwrite('result/{}_mask.jpg'.format(i), mask)
        #cv2.imwrite('temp/edge.jpg', edge)
        cv2.imwrite('result/{}_draw.jpg'.format(i), org)


if __name__ == '__main__':
    main2()