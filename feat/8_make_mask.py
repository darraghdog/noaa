#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 19:46:46 2017

@author: darragh
"""

import os
import pandas as pd
from matplotlib import pyplot as plt
import scipy
from shutil import copyfile
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from scipy.misc import imsave
from PIL import Image
import numpy as np
import warnings
import matplotlib.pyplot as plt
from skimage import draw
warnings.filterwarnings('ignore')

# 'adult_males', 'subadult_males','adult_females','juveniles','pups',
os.chdir('/home/darragh/Dropbox/noaa/feat')
boundaries = [100,80,70,70,30]
colors = ['red', 'blue', 'green', 'yellow', 'pink']
validate = False
validate_rfcn = False
bbimg_trn = False
bbimg_tst = False # this is the big one
bbimgblk_trn = False
annotations = False
boundary = 40
block_size = 544
img_w = 4896 # width
img_h = 3264 # height
RAW = '/hdd/ultra/raw'
ULTRA = '/hdd/ultra'
TESTDIR = '../data/Test'

def makeDir(name):
    if not os.path.exists(name):
        os.mkdir(name)

# Load up the coordinates
coords = pd.read_csv("block_coords.csv")
coords = coords[['id', 'block', 'class', 'block_width', 'block_height']]
img_files = coords[['id', 'block']].drop_duplicates().reset_index(drop=True)
os.chdir('/hdd/ultra')

'''
FIRST ALL FOR ONE CLASS
'''

# Process training images
makeDir('raw1')
img_all = os.listdir('JPEGImagesBlk')
for f in range(2):
    fold = 'fold'+str(f+1) 
    makeDir('raw1/'+fold)
    for c, row in img_files[img_files['id']%2==f].reset_index(drop=True).iterrows():
        if '%s_%s.jpg'%(row[0], row[1]) not in img_all : 
            continue
        dftmp = coords[(coords['id']==row['id']) & (coords['block']==row['block']) ].reset_index(drop=True)
        tif = np.zeros((544,544))
        for c1, row1 in dftmp.iterrows():
            radius = boundaries[row1['class']]/2
            rr, cc = draw.circle(row1['block_height'], row1['block_width'], radius=radius, shape=tif.shape)
            tif[rr, cc] = 1
        imsave('raw1/'+ fold +'/%s_%s.tif'%(row[0], row[1]), tif)
        src = os.path.join(ULTRA, 'JPEGImagesBlk/%s_%s.jpg'%(row[0], row[1]))
        dst = os.path.join(ULTRA, 'raw1', fold, '%s_%s.jpg'%(row[0], row[1]))
        copyfile(src, dst)

# Create a file with the test images 
preds = pd.read_csv('~/Dropbox/noaa/coords/vggTestPreds2604.csv')
idx = preds.groupby(['img'])['predSeal'].transform(max) == preds['predSeal']
preds = preds[idx]
preds = preds[preds['predSeal']>0.6]
preds['img'] = preds['img'] + '.jpg'
preds[['img']].to_csv(os.path.join(ULTRA, 'test_imgname.csv.gz'), 
     compression = "gzip", index = False)



arr = np.zeros((200, 200))
rr, cc = draw.circle(100, 100, radius=40, shape=arr.shape)
arr[rr, cc] = 1
plt.imshow(arr)
plt.show()

plt.imshow(tif)
plt.show()

'''
NOW MULTIPLE CLASSES
'''

# Process training images
boundaries = [90,80,60,60,40]
makeDir('raw2')
vggCVpred = pd.concat([pd.read_csv('~/Dropbox/noaa/coords/vggCVPreds2604_fold1.csv'),
           pd.read_csv('~/Dropbox/noaa/coords/vggCVPreds2604_fold2.csv')], axis = 0)
idx = vggCVpred.groupby(['img'])['predSeal'].transform(max) == vggCVpred['predSeal']
vggCVpred = vggCVpred[idx]
vggCVpred = vggCVpred[vggCVpred['predSeal']>0.6].reset_index(drop=True)
vggCVpred['id'] = vggCVpred['img'].map(lambda x: int(x.split('_')[0]))
vggCVpred['block'] = vggCVpred['img'].map(lambda x: x.split('_')[1])

img_all = os.listdir('JPEGImagesBlk')
for f in range(2):
    fold = 'fold'+str(f+1) 
    makeDir('raw2/'+fold)
    for c, row in vggCVpred[vggCVpred['id']%2==f].reset_index(drop=True).iterrows():
        tif = np.zeros((544,544, 6), dtype=np.uint8)
        if row['id'] in coords.id:
            dftmp = coords[(coords['id']==row['id']) & (coords['block']==int(row['block'])) ].reset_index(drop=True)
            for c1, row1 in dftmp.iterrows():
                radius = boundaries[row1['class']]/2
                rr, cc = draw.circle(row1['block_height'], row1['block_width'], radius=radius, shape=tif.shape)
                tif[rr, cc, row1['class']] = 1
        
        np.save('raw2/'+ fold +'/%s_%s'%(row[2], row[3]), tif)
        src = os.path.join(ULTRA, 'JPEGImagesBlk/%s_%s.jpg'%(row[2], row[3]))
        dst = os.path.join(ULTRA, 'raw2', fold, '%s_%s.jpg'%(row[2], row[3]))
        copyfile(src, dst)
