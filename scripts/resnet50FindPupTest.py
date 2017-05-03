import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
import xgboost as xgb
from matplotlib import pyplot as plt
import numpy as np
import gc, math
import pickle
import warnings
warnings.filterwarnings('ignore')

import scipy
from sklearn.metrics import fbeta_score
from PIL import Image
from scipy.ndimage import imread

from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

# Functions
def preprocess_input(x):
    #resnet50 image preprocessing
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    x[:, :, 0] -= 100
    x[:, :, 1] -= 115
    x[:, :, 2] -= 124
    return x

def create_rect5(row):
    if is_seal:
        return plt.Rectangle((row['x0'], row['y0']), row['w'], row['h'], color='red', fill=False, lw=2)
    else:
        return plt.Rectangle((row['x0'], row['y0']), row['w'], row['h'], color='red', fill=False, lw=4)

def load_img(path, bbox, target_size=None):
    img = Image.open(path)
    img = img.convert('RGB')
    cropped = img.crop((bbox[0],bbox[1],bbox[2],bbox[3]))
    if target_size:
        cropped = cropped.resize((target_size[1], target_size[0]))
    return cropped

# Params
validate = False
img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_class = 2
ROWS, COLS = 224, 224
BATCHSIZE = 32
PUP_CLASSES = ['NoP', 'pup']
nb_perClass = int(BATCHSIZE / len(PUP_CLASSES))
TRAIN_DIR = '../darknet/seals/JPEGImagesBlk'
TEST_DIR = '../darknet/seals/JPEGImagesTest'

# Load resnet50 predicted seals
os.chdir('/home/darragh/Dropbox/noaa/feat')
resnet50_train = pd.concat([pd.read_pickle('../coords/resnet50CVPreds2604_fold1.pkl'),
                       pd.read_pickle('../coords/resnet50CVPreds2604_fold2.pkl')],axis=0)
resnet50_train = resnet50_train[resnet50_train['predSeal']>0.6].reset_index(drop=True)
resnet50_train = resnet50_train.drop(['proba','predNoSeal', 'predSeal'], axis = 1)

resnet50_test = pd.concat([pd.read_pickle('../coords/rfcnTst.pkl'),
                      pd.read_csv('../coords/resnet50TestPreds2604.csv')[['predSeal']]],axis=1)
resnet50_test = resnet50_test[resnet50_test['predSeal']>0.6].reset_index(drop=True)
resnet50_test = resnet50_test.drop(['proba', 'predSeal'], axis = 1)

# Loads the blocks and pull out the pups
block_coords = pd.read_csv('../coords/block_coords.csv')
block_coords['block'] = block_coords['block'].map(str).apply(lambda x: '{0:0>2}'.format(x))
block_coords['img'] = block_coords['id'].map(str) + '_' + block_coords['block'].map(str)
pup_coords = block_coords[block_coords['class']==4][['img', 'block_width', 'block_height']].reset_index(drop=True)

# Add to resnet50_train the images with a pup
resnet50_train['with_pup'] = 0
border = 5  # Make sure the pup is inside the image by 5 pixels
for pup_img in tqdm(pup_coords.img.unique(), miniters=20):
    if pup_img in resnet50_train.img.values:
        resnet50_tmp = resnet50_train[resnet50_train['img'] == pup_img]
        pup_tmp = pup_coords[pup_coords['img'] == pup_img]
        for c, row in resnet50_tmp.iterrows():
            if ((pup_tmp['block_width'] >= (row['x0']+border)) & (pup_tmp['block_width'] <= (row['x1']-border)) & \
            (pup_tmp['block_height'] >= (row['y0']+border)) & (pup_tmp['block_height'] <= (row['y1']-border))).any():
                resnet50_train.loc[c, 'with_pup'] = 1
resnet50_train.with_pup.hist()
resnet50_train.seal.hist()

# Now lets output and see if we can predict if we have a pup or not. 
resnet50_train.to_pickle('../coords/resnet_pups_trn.pkl')
resnet50_test.to_pickle('../coords/resnet_pups_tst.pkl')

# Lets validate the train file
if validate:
    samp = '792_23'
    cond = resnet50_train.img.str.contains(samp)
    for img_name in resnet50_train[cond].img.unique():
        img = imread('../data/JPEGImagesBlk/%s.jpg'%(img_name))
        bbox = resnet50_train[resnet50_train['img'] == img_name]
        bbox['w'] = bbox['x1'] - bbox['x0']
        bbox['h'] = bbox['y1'] - bbox['y0']
        plt.figure(figsize=(20,20))
        plt.imshow(img)
        for c, row in bbox.iterrows():
            plt.gca().add_patch(plt.Rectangle((row['x0'], row['y0']), row['w'],\
            row['h'], color='red', fill=False, lw=1+(5*row['with_pup'])))
            
# Now lets model...
def train_generator(datagen, df):
    while 1:
        batch_x = np.zeros((BATCHSIZE, ROWS, COLS, 3), dtype=K.floatx())
        batch_y = np.zeros((BATCHSIZE, len(PUP_CLASSES)), dtype=K.floatx())
        fn = lambda obj: obj.loc[np.random.choice(obj.index, size=nb_perClass, replace=False),:]
        batch_df = df.groupby(['with_pup'], as_index=True).apply(fn)
        i = 0
        for index,row in batch_df.iterrows():
            row = row.tolist()
            image_file = os.path.join(TRAIN_DIR, row[0])
            with_pup = row[6]
            bbox = row[2:6]
            cropped = load_img(image_file+'.jpg',bbox,target_size=(ROWS,COLS))
            x = np.asarray(cropped, dtype=K.floatx())
            x = datagen.random_transform(x)
            x = preprocess_input(x)
            batch_x[i] = x
            batch_y[i,pup] = 1
            i += 1
        yield (batch_x.transpose(0, 3, 1, 2), batch_y)

def test_generator(df, datagen = None, batch_size = BATCHSIZE):
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
        batch_x = np.zeros((batch_df.shape[0], ROWS, COLS, 3), dtype=K.floatx())
        i = 0
        for index,row in batch_df.iterrows():
            row = row.tolist()
            image_file = os.path.join(TEST_DIR, row[0]+'.jpg')
            bbox = row[2:6]
            cropped = load_img(image_file,bbox,target_size=(ROWS,COLS))
            x = np.asarray(cropped, dtype=K.floatx())
            if datagen is not None: x = datagen.random_transform(x)            
            x = preprocess_input(x)
            batch_x[i] = x
            i += 1
        if batch_index%50 == 0: print(batch_index)
        #return(batch_x.transpose(0, 3, 1, 2))
        yield(batch_x.transpose(0, 3, 1, 2))
        
# Data generator
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True)
    
# Lets make our validation set
CVsplit = resnet50_train.img.str.split('_').apply(lambda x: x[0]).astype(int) % 10 == 0
train_df = resnet50_train[~CVsplit]
valid_df = resnet50_train[CVsplit]
test_df = resnet50_test

# validation_data (valid_x,valid_y)
df_1 = valid_df
l = valid_df.groupby('with_pup').size()
nb_NoF_valid = math.ceil(l.sum()/10)
valid_x = np.zeros((valid_df.shape[0], ROWS, COLS, 3), dtype=K.floatx())
valid_y = np.zeros((valid_df.shape[0], len(PUP_CLASSES)), dtype=K.floatx())
i = 0
for index,row in valid_df.iterrows():
    row = row.tolist()
    image_file = os.path.join(TRAIN_DIR, row[0])
    with_pup = row[6]
    bbox = row[2:6]
    cropped = load_img(image_file+'.jpg',bbox,target_size=(ROWS,COLS))
    x = np.asarray(cropped, dtype=K.floatx())
    x = preprocess_input(x)
    valid_x[i] = x
    valid_y[i,pup] = 1
    i += 1
valid_x = valid_x.transpose(0, 3, 1, 2)
valid_x.shape

# Now lets see how our model works
# Load our model
nb_epoch = 2
samples_per_epoch = 40000
model = resnet50_model(ROWS, COLS, channel, num_class)
for layer in model.layers:
    layer.trainable = False
for layer in model.layers[-3:]:
    layer.trainable = True

# Start Fine-tuning
model.fit_generator(train_generator(train_datagen, train_df),
          nb_epoch=nb_epoch,
          samples_per_epoch=samples_per_epoch, #50000,
          verbose=1,
          validation_data=(valid_x, valid_y),
          )

for layer in model.layers[38:]:
    layer.trainable = True
model.optimizer.lr = 1e-5
nb_epoch = 6
model.fit_generator(train_generator(train_datagen, df=train_df),
          nb_epoch=nb_epoch,
          samples_per_epoch=samples_per_epoch,
          verbose=1,
          validation_data=(valid_x, valid_y),
          )

# test_preds = test_model.predict_generator(test_generator(test_df), val_samples=test_df.shape[0]))
test_preds = model.predict_generator(test_generator(test_df), val_samples=test_df.shape[0])

df = pd.concat([test_df, pd.DataFrame(test_preds,  columns=['predNoPup', 'predPup'])], axis=1)
df.to_pickle('../coords/resnet50TestPredsPups_0105.pkl')

df[['img', 'predPup']].to_csv('../coords/resnet50TestPredsPups_0105.csv', index=False)
df.head(5)

model.summary()

model.save('checkpoints/model_resnet50TestPreds_Pups_0105.h5')