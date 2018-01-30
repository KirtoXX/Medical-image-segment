import SimpleITK as sitk
from PIL import Image
#import pydicom
import numpy as np
import cv2
import os
from scipy import misc


def loadFile(filename):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    frame_num, width, height = img_array.shape
    print(img_array.dtype)
    return img_array, frame_num, width, height

#aruguement
def limitedEqualize(img_array, limit=2.0):
    img_array_list = []
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))
    for img in img_array:
        img = img.astype(dtype=np.uint8)
        img_array_list.append(clahe.apply(img))
    return img_array_list

def process_img(path,dir=None):
    images1, frame_num, width, height = loadFile(path)
    images = limitedEqualize(images1, limit=0)
    images = np.expand_dims(images, axis=-1)
    os.makedirs('data/'+str(dir)+'/')
    for i,img in enumerate(images):
        filename = 'data/img/'+str(dir)+'/'+str(i)+'.jpg'
        print(filename)
        cv2.imwrite(filename,img)
    print('done')

def process_mask(path,dir=None):
    image, frame_num, _, _ = loadFile(path)
    for i,img in enumerate(image):
        img = img.astype(dtype=np.float32)
        img = img*255
        filename = 'data/seg/' + str(dir) + '/' + str(i) + '.jpg'
        cv2.imwrite(filename,img)
    print('done')

def main1():
    path = 'f:/dicom_data/volume-0.nii'
    images1,frame_num, width, height = loadFile(path)
    print(frame_num,width,height)

    images = limitedEqualize(images1,limit=0)
    images = np.expand_dims(images,axis=-1)
    print(len(images))
    print(images[0].shape)

    temp = images[0]
    temp = temp.astype(dtype=np.uint8)
    cv2.imshow('image',temp)
    cv2.waitKey(0)

def main2():
    path = 'f:/dicom_data/train/seg/'
    img_dir = os.listdir(path)
    for i,dicom in enumerate(img_dir):
        print(i)
        path2 = path+dicom
        process_mask(path2,dir=i)


def main3():
    img_path = 'f:/dicom_data/train/img/'
    seg_path = 'f:/dicom_data/train/seg/'

    img_save = 'data/img/'
    seg_save = 'data/seg/'

    file_name = os.listdir(img_path)
    for j in range(len(file_name)):
        file = 'volume-'+str(j)+'.nii'
        image_file_path = img_save+file
        seg_file_path = seg_save+'segmentation-'+str(j)+'.nii'
        #build dir
        os.makedirs(image_file_path)
        os.makedirs(seg_file_path)
        images, frame_num, width, height = loadFile(img_path+file)
        #enhance the img
        images = limitedEqualize(images,limit=4.)
        segs,_,_,_ = loadFile(seg_path+'segmentation-'+str(j)+'.nii')
        segs = segs.astype(np.float32)*255.
        #save img seg
        for i in range(frame_num):
            # define the size of seg
            if np.sum(segs[i]==255.)>10000:
                image_temp = images[i]
                seg_temp = segs[i]
                cv2.imwrite(image_file_path+'/'+str(i)+'.jpg',image_temp)
                cv2.imwrite(seg_file_path+'/'+str(i)+'.jpg',seg_temp)
        print(file)



if __name__ == '__main__':
    main3()






