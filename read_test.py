import numpy as np
import cv2
from NII_to_jpeg import loadFile,limitedEqualize

def main():
    path = 'F:/dicom_data/train/img/volume-56.nii'
    img,_,_,_ = loadFile(path)
    img = img.astype(dtype=np.uint8)
    #img = limitedEqualize(img,limit=0.0)
    image = img[56]
    cv2.imshow('img',image)
    img2 = cv2.imread('data/img/volume-56.nii/195.jpg')
    print(img2.dtype)
    cv2.imshow('img2',img2)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
