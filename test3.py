import numpy as np
import cv2

def get_s(path):
    img = cv2.imread(path)
    s = np.sum(img==255)
    return s

def main():
    path = 'data/seg/segmentation-26.nii/276.jpg'
    s = get_s(path)
    print(s)


if __name__ == '__main__':
    main()