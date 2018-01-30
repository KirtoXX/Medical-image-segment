import tensorflow as tf
import os

def path_to_image(dir):
    file_names = os.listdir(dir)
    imgs = []
    for image in file_names:
        imgs.append(dir+"/"+image)
    return imgs

def get_path():
    img_dir = 'data/img/'
    seg_dir = 'data/seg/'
    file_dir = os.listdir(img_dir)
    img = []
    seg = []
    for i,name in enumerate(file_dir):
        img_path = img_dir+'volume-'+str(i)+'.nii'
        seg_path = seg_dir+'segmentation-'+str(i)+'.nii'
        data1 = path_to_image(img_path)
        data2 = path_to_image(seg_path)
        img.extend(data1)
        seg.extend(data2)
    print('list make finsih')
    return img,seg

def processing(img):
    img_string = tf.read_file(img)
    #img_decoded = tf.image.decode_image(img_string,channels=1)  # uint8 range from 1~255
    img_decoded = tf.image.decode_jpeg(img_string,channels=1)
    img_decoded = tf.image.resize_images(img_decoded,[512,512])
    return img_decoded

def _parse_function(img,seg):
  img_decoded = processing(img)
  seg_decoded = processing(seg)
  seg_decoded = tf.image.resize_images(seg_decoded,[128,128])
  seg_decoded = tf.cast(seg_decoded,tf.float32)/255.
  return img_decoded,seg_decoded

def file_to_tensor(epoch=100,batch_size=32):
    img, seg = get_path()
    #buffer_size = len(img)
    dataset = tf.data.Dataset.from_tensor_slices((img,seg))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    img,seg = iterator.get_next()
    return img,seg

def main():
    img,seg = get_path()
    print(img[1000])
    print(seg[1000])

if __name__ == '__main__':
    main()

