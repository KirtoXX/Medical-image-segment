from read_data import file_to_tensor
import tensorflow as tf
import cv2

#test
def main():
    img,seg = file_to_tensor()
    with tf.Session() as sess:
        img1,seg1 = sess.run([img,seg])
        print(img1.shape,img1.dtype)
        img2 = cv2.resize(seg1[3],(512,512),interpolation=cv2.INTER_AREA)
        cv2.imshow('seg',img2)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()

