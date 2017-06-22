import os
import numpy as np
from keras import backend as K
from keras.layers import Input, MaxPooling2D, UpSampling2D, Conv2D
#from keras.layers import concatenate
#from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from skimage.transform import resize
from skimage.io import imsave
#from seg_noaa import load_train_data, load_test_data
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from skimage.io import imsave, imread
from skimage.transform import resize 
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from keras import backend as K

# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1
smooth = 1.



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def double_conv_layer(x, size, dropout, batch_norm):
    conv = Convolution2D(size, 3, 3, border_mode='same')(x)
    if batch_norm == True:
        conv = BatchNormalization(mode=0, axis=1)(conv)
    conv = Activation('relu')(conv)
    conv = Convolution2D(size, 3, 3, border_mode='same')(conv)
    if batch_norm == True:
        conv = BatchNormalization(mode=0, axis=1)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    return conv

def create_model(img_rows, img_cols, img_channels, smooth = 1., dropout_val=0.05, batch_norm=True):
    inputs = Input((INPUT_CHANNELS, img_rows, img_cols))
    conv1 = double_conv_layer(inputs, 32, dropout_val, batch_norm)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = double_conv_layer(pool1, 64, dropout_val, batch_norm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv_layer(pool2, 128, dropout_val, batch_norm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = double_conv_layer(pool3, 256, dropout_val, batch_norm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = double_conv_layer(pool4, 512, dropout_val, batch_norm)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = double_conv_layer(pool5, 1024, dropout_val, batch_norm)

    up6 = merge([UpSampling2D(size=(2, 2))(conv6), conv5], mode='concat', concat_axis=1)
    conv7 = double_conv_layer(up6, 512, dropout_val, batch_norm)

    up7 = merge([UpSampling2D(size=(2, 2))(conv7), conv4], mode='concat', concat_axis=1)
    conv8 = double_conv_layer(up7, 256, dropout_val, batch_norm)

    up8 = merge([UpSampling2D(size=(2, 2))(conv8), conv3], mode='concat', concat_axis=1)
    conv9 = double_conv_layer(up8, 128, dropout_val, batch_norm)

    up9 = merge([UpSampling2D(size=(2, 2))(conv9), conv2], mode='concat', concat_axis=1)
    conv10 = double_conv_layer(up9, 64, dropout_val, batch_norm)

    up10 = merge([UpSampling2D(size=(2, 2))(conv10), conv1], mode='concat', concat_axis=1)
    conv11 = double_conv_layer(up10, 32, 0, batch_norm)

    conv12 = Convolution2D(OUTPUT_MASK_CHANNELS, 1, 1)(conv11)
    conv12 = BatchNormalization(mode=0, axis=1)(conv12)
    conv12 = Activation('sigmoid')(conv12)

    model = Model(input=inputs, output=conv12)
    return model

def preprocess_img(imgs, img_rows, img_cols, img_channels):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, img_channels), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i] = imgs[i]

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def preprocess(imgs, img_rows, img_cols, img_channels):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, img_channels), dtype=np.uint8)
    for i in range(imgs.shape[0]):
    #    imgs_p[i] = resize(imgs[i], (img_cols, img_rows, 1), preserve_range=True)
        imgs_p[i][:,:,0] = imgs[i]
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def test_generator(df, img_rows, img_cols, data_path, test_path, batch_size = 4):
    n = df.shape[0]
    batch_index = 0
    while 1:
        current_index = batch_index * batch_size
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1    
        else:
            current_batch_size = n - current_index
            batch_index = 0        
        batch_df = df[current_index:current_index+current_batch_size]
        batch_x = np.zeros((batch_df.shape[0], img_rows, img_cols, 3)).astype('float32')
        i = 0
        for index,row in batch_df.iterrows():
            img = imread(os.path.join(data_path, test_path, row[0]), as_grey=False)
            img = resize(img, (img_rows, img_cols), mode='reflect')
            x = np.array([img])
            x -= mean
            x /= std
            batch_x[i] = x
            i += 1
        if batch_index%300 == 0: print(batch_index)
        yield(batch_x.transpose(0, 3, 1, 2))
        #return batch_x
