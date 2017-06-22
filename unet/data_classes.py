from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread
from skimage.transform import resize 

data_path = 'raw2/'

image_rows = 544/2
image_cols = 544/2
classes = 6

def multi_resize(img_mask, image_rows=image_rows, image_cols=image_cols, classes=classes):
    imout = np.ndarray((image_rows, image_cols, classes), dtype=np.uint8)
    for i in range(classes):
        imout[:,:,i] = resize(img_mask[:,:,i].astype(np.float32), (image_rows, image_cols), mode='reflect')
    return imout

def create_train_data(classes=6):
    # train_data_path = os.path.join(data_path, 'train')
    train_data_path = os.path.join(data_path, 'fold1')
    images = os.listdir(train_data_path)
    total = len(images) / 2

    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.float32)
    imgs_mask = np.ndarray((total, image_rows, image_cols, classes), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'npy' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '.npy'
        img = imread(os.path.join(train_data_path, image_name), as_grey=False)
        #img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)
        img_mask = np.load(os.path.join(train_data_path, image_mask_name))
        img = resize(img, (image_rows, image_cols), mode='reflect')
        img_mask = multi_resize(img_mask)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 500 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train_noaa_class.npy', imgs)
    np.save('imgs_mask_train_noaa_class.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train_noaa_class.npy')
    imgs_mask_train = np.load('imgs_mask_train_noaa_class.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    #train_data_path = os.path.join(data_path, 'test')
    train_data_path = os.path.join(data_path, 'fold2')
    images = os.listdir(train_data_path)
    total = len(images)/2

    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.float32)
    # imgs_id = np.ndarray((total, ), dtype=np.int32)
    imgs_id = []

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        if 'npy' in image_name:
            continue
        img_id = image_name.split('.')[0]
        img = imread(os.path.join(train_data_path, image_name), as_grey=False)
        img = resize(img, (image_rows, image_cols), mode='reflect')

        img = np.array([img])

        imgs[i] = img
        # imgs_id[i] = img_id
        imgs_id.append(img_id)

        if i % 500 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test_noaa_class.npy', imgs)
    np.save('imgs_id_test_noaa_class.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test_noaa_class.npy')
    imgs_id = np.load('imgs_id_test_noaa_class.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_train_data()
    create_test_data()