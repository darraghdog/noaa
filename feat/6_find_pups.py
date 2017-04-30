import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
import xgboost as xgb
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import scipy
from sklearn.metrics import fbeta_score
from PIL import Image
from scipy.ndimage import imread

validate = True

# Load VGG predicted seals
os.chdir('/home/darragh/Dropbox/noaa/feat')
vgg_train = pd.concat([pd.read_pickle('../coords/vggCVPreds2604_fold1.pkl'),
                       pd.read_pickle('../coords/vggCVPreds2604_fold2.pkl')],axis=0)
vgg_train = vgg_train[vgg_train['predSeal']>0.6].reset_index(drop=True)
vgg_train = vgg_train.drop(['proba','predNoSeal', 'predSeal'], axis = 1)

vgg_test = pd.concat([pd.read_pickle('../coords/rfcnTst.pkl'),
                      pd.read_csv('../coords/vggTestPreds2604.csv')[['predSeal']]],axis=1)
vgg_test = vgg_test[vgg_test['predSeal']>0.6].reset_index(drop=True)
vgg_test = vgg_test.drop(['proba', 'predSeal'], axis = 1)

# Loads the blocks and pull out the pups
block_coords = pd.read_csv('../coords/block_coords.csv')
block_coords['block'] = block_coords['block'].map(str).apply(lambda x: '{0:0>2}'.format(x))
block_coords['img'] = block_coords['id'].map(str) + '_' + block_coords['block'].map(str)
pup_coords = block_coords[block_coords['class']==4][['img', 'block_width', 'block_height']].reset_index(drop=True)

# Add to vgg_train the images with a pup
vgg_train['with_pup'] = 0
for pup_img in pup_coords.img.unique():
    if pup_img in vgg_train.img.values:
        vgg_tmp = vgg_train[vgg_train['img'] == pup_img]
        pup_tmp = pup_coords[pup_coords['img'] == pup_img]
        for c, row in vgg_tmp.iterrows():
            if ((pup_tmp['block_width'] >= row['x0']) & (pup_tmp['block_width'] <= row['x1']) & \
            (pup_tmp['block_height'] >= row['y0']) & (pup_tmp['block_height'] <= row['y1'])).any():
                vgg_train.loc[c, 'with_pup'] = 1

# Lets validate the train file
if validate:
    samp = '945_05'
    cond = vgg_train.img.str.contains(samp)
    for img_name in vgg_train[cond].img.unique():
        img = imread('../data/JPEGImagesBlk/%s.jpg'%(img_name))
        bbox = vgg_train[vgg_train['img'] == img_name]
        bbox['w'] = bbox['x1'] - bbox['x0']
        bbox['h'] = bbox['y1'] - bbox['y0']
        plt.figure(figsize=(20,20))
        plt.imshow(img)
        for c, row in bbox.iterrows():
            plt.gca().add_patch(plt.Rectangle((row['x0'], row['y0']), row['w'],\
            row['h'], color='red', fill=False, lw=1+(5*row['with_pup'])))

