# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:27:44 2017
This was changed on May 1st to get out more images to send through a classifier

@author: darragh
"""
import os, gc
import pandas as pd
from matplotlib import pyplot as plt
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from scipy.misc import imsave
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm, tqdm_pandas
tqdm_pandas(tqdm())

# 'adult_males', 'subadult_males','adult_females','juveniles','pups',
os.chdir('/home/darragh/Dropbox/noaa/feat')
boundaries = [100,80,70,70,40]
colors = ['red', 'blue', 'green', 'yellow', 'pink']
make_train = False
make_train_classes = False
make_test = True
validate_train = False
validate_test = False
cutoff = 0.7
block_size = 544
img_w = 4896 # width
img_h = 3264 # height
boundaries = [100,80,70,70,40]
colors = ['red', 'blue', 'green', 'yellow', 'pink']
check_border = 20
SEAL_CLASSES = ['NoS', 'Seal', 'Other']
ROWS = 100
COLS = 100
TESTDIR = '../data/Test'

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

# For each row, check if there is a seal in the block
def is_seal(row):
    row_id, row_block = row['img'].split('_')
    seal = ((coords['id']==int(row_id)) & \
                (coords['block'] == row_block) & \
                (coords['block_width']>(int(row['x0'])-check_border)) & \
                (coords['block_width']<(int(row['x1'])+check_border)) & \
                (coords['block_height']>(int(row['y0'])-check_border)) & \
                (coords['block_height']<(int(row['y1'])+check_border))).any()
    return int(seal)

# Load the Xtrain files
if make_train:
    # Get the ground truth labels
    coords = pd.read_csv("../feat/coords_meta.csv")
    train_meta = pd.read_csv("../feat/train_meta.csv", usecols = ['id', 'height', 'width', 'all_diff'])#,\
    train_meta.columns = ['id', 'img_height', 'img_width', 'all_diff']
    coords = pd.merge(coords, train_meta, on='id', how='inner')
    coords['block_width'] = coords['width'].apply(lambda x: x*img_w).div(coords['img_width'], axis=0).apply(int)%block_size
    coords['block_height'] = coords['height'].apply(lambda x: x*img_h).div(coords['img_height'], axis=0).apply(int)%block_size
    coords['block'] = coords['width'].apply(lambda x: x*img_w).div(coords['img_width'], axis=0).apply(int).apply(lambda x: x/block_size).apply(str)+\
                        coords['height'].apply(lambda x: x*img_h).div(coords['img_height'], axis=0).apply(int).apply(lambda x: x/block_size).apply(str)

    def make_CV_boxes(cutoff1, file_name):
        rfcnCVfold2 = pd.read_csv("../coords/comp4_30000_det_test_seals_fold2.txt",\
                            delimiter = " ", header=None, names=['img', 'proba', 'x0', 'y0', 'x1', 'y1'])
        rfcnCVfold1 = pd.read_csv("../coords/comp4_30000_det_test_seals_fold1.txt",\
                            delimiter = " ", header=None, names=['img', 'proba', 'x0', 'y0', 'x1', 'y1'])
        rfcnCVfold1['img'] = rfcnCVfold1['img'].str.replace('/home/ubuntu/noaa/darknet/seals/JPEGImagesBlk/', '')
        rfcnCV = pd.concat([rfcnCVfold2, rfcnCVfold1])
        rfcnCV = rfcnCV[rfcnCV['proba']>cutoff1]
        rfcnCV = rfcnCV[(rfcnCV['x1']-rfcnCV['x0'])<150]
        rfcnCV = rfcnCV[(rfcnCV['y1']-rfcnCV['y0'])<150]
        del rfcnCVfold1, rfcnCVfold2
        gc.collect()
        rfcnCV['seal'] = rfcnCV.progress_apply(is_seal, axis=1)
        rfcnCV = rfcnCV.reset_index(drop=True)
        rfcnCV['h_diff'] = ROWS - (rfcnCV['y1'] -rfcnCV['y0'])
        rfcnCV['w_diff'] = COLS - (rfcnCV['x1'] -rfcnCV['x0'])
        rfcnCV[rfcnCV['h_diff']<0]['h_diff'] = 0
        rfcnCV[rfcnCV['w_diff']<0]['w_diff'] = 0
        rfcnCV['x0'] = rfcnCV['x0'] - rfcnCV['w_diff'].divide(2)
        rfcnCV['x1'] = rfcnCV['x1'] + rfcnCV['w_diff'].divide(2)
        rfcnCV['y0'] = rfcnCV['y0'] - rfcnCV['h_diff'].divide(2)
        rfcnCV['y1'] = rfcnCV['y1'] + rfcnCV['h_diff'].divide(2)
        rfcnCV[['x0', 'x1']] = rfcnCV[['x0', 'x1']].add(np.where(rfcnCV['x0']<0, rfcnCV['x0'].abs(), 0), axis = 0 )
        rfcnCV[['y0', 'y1']] = rfcnCV[['y0', 'y1']].add(np.where(rfcnCV['y0']<0, rfcnCV['y0'].abs(), 0), axis = 0 )
        rfcnCV[['x0', 'x1']] = rfcnCV[['x0', 'x1']].subtract(np.where(rfcnCV['x1']>block_size, (rfcnCV['x1']-block_size).abs(), 0), axis = 0 )
        rfcnCV[['y0', 'y1']] = rfcnCV[['y0', 'y1']].subtract(np.where(rfcnCV['y1']>block_size, (rfcnCV['y1']-block_size).abs(), 0), axis = 0 )
        rfcnCV.drop(['h_diff', 'w_diff'], axis=1, inplace=True)
        #rfcnCV.to_pickle('../coords/rfcnCV.pkl')
        rfcnCV.to_pickle(file_name)
        return rfcnCV
    rfcnCV = make_CV_boxes(cutoff, '../coords/rfcnCV.pkl')
    rfcnCVlo06 = make_CV_boxes(0.6, '../coords/rfcnCVlo06.pkl')
else:
    rfcnCV = pd.read_pickle('../coords/rfcnCV.pkl' )
    rfcnCVlo06 = pd.read_pickle('../coords/rfcnCVlo06.pkl' )

# For each row, check if there is a seal in the block
def seal_class(row):
    row_id, row_block = row['img'].split('_')
    # We have an entry in coords for that box
    check_border = 0
    box_classes = [0, 0, 0, 0, 0, 0]
    if ((coords['id']==int(row_id)) & \
                (coords['block'] == row_block) & \
                (coords['block_width']>(int(row['x0'])-check_border)) & \
                (coords['block_width']<(int(row['x1'])+check_border)) & \
                (coords['block_height']>(int(row['y0'])-check_border)) & \
                (coords['block_height']<(int(row['y1'])+check_border))).any():
        dfclass = coords[(coords['id']==int(row_id)) & \
                (coords['block'] == row_block) & \
                (coords['block_width']>(int(row['x0'])-check_border)) & \
                (coords['block_width']<(int(row['x1'])+check_border)) & \
                (coords['block_height']>(int(row['y0'])-check_border)) & \
                (coords['block_height']<(int(row['y1'])+check_border))]
        for c, irow in dfclass.iterrows():
            box_classes[irow['class']] = 1
        return box_classes
    else:
        # if it's not a seal mark the 'Not seal' class
        return [0, 0, 0, 0, 0, 1]


# Here instead of a binary label of a seal or not, we do a multiclass label so see which class of seal it is
if make_train_classes:
    # Get the ground truth labels
    coords = pd.read_csv("../feat/coords_meta.csv")
    train_meta = pd.read_csv("../feat/train_meta.csv", usecols = ['id', 'height', 'width', 'all_diff'])#,\
    train_meta.columns = ['id', 'img_height', 'img_width', 'all_diff']
    coords = pd.merge(coords, train_meta, on='id', how='inner')
    coords['block_width'] = coords['width'].apply(lambda x: x*img_w).div(coords['img_width'], axis=0).apply(int)%block_size
    coords['block_height'] = coords['height'].apply(lambda x: x*img_h).div(coords['img_height'], axis=0).apply(int)%block_size
    coords['block'] = coords['width'].apply(lambda x: x*img_w).div(coords['img_width'], axis=0).apply(int).apply(lambda x: x/block_size).apply(str)+\
                        coords['height'].apply(lambda x: x*img_h).div(coords['img_height'], axis=0).apply(int).apply(lambda x: x/block_size).apply(str)

    def make_CV_class_boxes(cutoff, file_name):
        rfcnCVfold2 = pd.read_csv("../coords/comp4_30000_det_test_seals_fold2.txt",\
                            delimiter = " ", header=None, names=['img', 'proba', 'x0', 'y0', 'x1', 'y1'])
        rfcnCVfold1 = pd.read_csv("../coords/comp4_30000_det_test_seals_fold1.txt",\
                            delimiter = " ", header=None, names=['img', 'proba', 'x0', 'y0', 'x1', 'y1'])
        rfcnCVfold1['img'] = rfcnCVfold1['img'].str.replace('/home/ubuntu/noaa/darknet/seals/JPEGImagesBlk/', '')
        rfcnCV = pd.concat([rfcnCVfold2, rfcnCVfold1])
        rfcnCV = rfcnCV[rfcnCV['proba']>cutoff]
        rfcnCV = rfcnCV[(rfcnCV['x1']-rfcnCV['x0'])<150]
        rfcnCV = rfcnCV[(rfcnCV['y1']-rfcnCV['y0'])<150]
        del rfcnCVfold1, rfcnCVfold2
        gc.collect()
    
        seal_classes = []
        for c, row in rfcnCV.reset_index(drop=True).iterrows():
            if c % 1000 == True: print ('Row ' + str(c))
            seal_classes.append(seal_class(row))
        rfcnCV['seal'] = rfcnCV.progress_apply(is_seal, axis=1)
        headers = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups', 'other']
        rfcnCV = pd.concat([rfcnCV.reset_index(drop=True), pd.DataFrame(seal_classes, columns=headers)], axis = 1)
        
        rfcnCV['h_diff'] = ROWS - (rfcnCV['y1'] -rfcnCV['y0'])
        rfcnCV['w_diff'] = COLS - (rfcnCV['x1'] -rfcnCV['x0'])
        rfcnCV[rfcnCV['h_diff']<0]['h_diff'] = 0
        rfcnCV[rfcnCV['w_diff']<0]['w_diff'] = 0
        rfcnCV['x0'] = rfcnCV['x0'] - rfcnCV['w_diff'].divide(2)
        rfcnCV['x1'] = rfcnCV['x1'] + rfcnCV['w_diff'].divide(2)
        rfcnCV['y0'] = rfcnCV['y0'] - rfcnCV['h_diff'].divide(2)
        rfcnCV['y1'] = rfcnCV['y1'] + rfcnCV['h_diff'].divide(2)
        rfcnCV[['x0', 'x1']] = rfcnCV[['x0', 'x1']].add(np.where(rfcnCV['x0']<0, rfcnCV['x0'].abs(), 0), axis = 0 )
        rfcnCV[['y0', 'y1']] = rfcnCV[['y0', 'y1']].add(np.where(rfcnCV['y0']<0, rfcnCV['y0'].abs(), 0), axis = 0 )
        rfcnCV[['x0', 'x1']] = rfcnCV[['x0', 'x1']].subtract(np.where(rfcnCV['x1']>block_size, (rfcnCV['x1']-block_size).abs(), 0), axis = 0 )
        rfcnCV[['y0', 'y1']] = rfcnCV[['y0', 'y1']].subtract(np.where(rfcnCV['y1']>block_size, (rfcnCV['y1']-block_size).abs(), 0), axis = 0 )
        rfcnCV.drop(['h_diff', 'w_diff'], axis=1, inplace=True)
        #rfcnCV.to_pickle('../coords/rfcnCV.pkl')
        rfcnCV.to_pickle(file_name)
        return rfcnCV
    rfcnCVclass = make_CV_class_boxes(cutoff, '../coords/rfcnCVclass.pkl')
    #rfcnCVlo06 = make_CV_boxes(0.6, '../coords/rfcnCVlo06.pkl')
else:
    rfcnCVclass = pd.read_pickle('../coords/rfcnCVclass.pkl' )
    #rfcnCVlo06 = pd.read_pickle('../coords/rfcnCVlo06.pkl' )

# Load the Xtrain files
if make_test:
    def make_Tst_boxes(cutoff1, file_name):
        rfcnTst = pd.read_csv("../coords/comp4_30000_det_test_all_seals_subset.txt", skiprows=1, \
                            delimiter = ",", header=None, names=['img', 'proba', 'x0', 'y0', 'x1', 'y1'])
        rfcnTst['img'] = rfcnTst['img'].str.replace('/home/ubuntu/noaa/darknet/seals/JPEGImagesTest/', '')
        rfcnTst = rfcnTst[rfcnTst['proba']>cutoff1]
        rfcnTst = rfcnTst[(rfcnTst['x1']-rfcnTst['x0'])<150]
        rfcnTst = rfcnTst[(rfcnTst['y1']-rfcnTst['y0'])<150]
        gc.collect()
        rfcnTst = rfcnTst.reset_index(drop=True)
        rfcnTst['h_diff'] = ROWS - (rfcnTst['y1'] -rfcnTst['y0'])
        rfcnTst['w_diff'] = COLS - (rfcnTst['x1'] -rfcnTst['x0'])
        rfcnTst[rfcnTst['h_diff']<0]['h_diff'] = 0
        rfcnTst[rfcnTst['w_diff']<0]['w_diff'] = 0
        rfcnTst['x0'] = rfcnTst['x0'] - rfcnTst['w_diff'].divide(2)
        rfcnTst['x1'] = rfcnTst['x1'] + rfcnTst['w_diff'].divide(2)
        rfcnTst['y0'] = rfcnTst['y0'] - rfcnTst['h_diff'].divide(2)
        rfcnTst['y1'] = rfcnTst['y1'] + rfcnTst['h_diff'].divide(2)
        rfcnTst[['x0', 'x1']] = rfcnTst[['x0', 'x1']].add(np.where(rfcnTst['x0']<0, rfcnTst['x0'].abs(), 0), axis = 0 )
        rfcnTst[['y0', 'y1']] = rfcnTst[['y0', 'y1']].add(np.where(rfcnTst['y0']<0, rfcnTst['y0'].abs(), 0), axis = 0 )
        rfcnTst[['x0', 'x1']] = rfcnTst[['x0', 'x1']].subtract(np.where(rfcnTst['x1']>block_size, (rfcnTst['x1']-block_size).abs(), 0), axis = 0 )
        rfcnTst[['y0', 'y1']] = rfcnTst[['y0', 'y1']].subtract(np.where(rfcnTst['y1']>block_size, (rfcnTst['y1']-block_size).abs(), 0), axis = 0 )
        rfcnTst.drop(['h_diff', 'w_diff'], axis=1, inplace=True)
        rfcnTst.to_pickle(file_name)
        return rfcnTst
    rfcnTst = make_Tst_boxes(cutoff, '../coords/rfcnTst.pkl')
    rfcnTstlo06 = make_Tst_boxes(0.6, '../coords/rfcnTstlo06.pkl')
else:
    rfcnTst = pd.read_pickle('../coords/rfcnTst.pkl')
    rfcnTstlo06 = pd.read_pickle('../coords/rfcnTstlo06.pkl')
 


# Lets validate the train file
if validate_test:
    cond = rfcnCV.img.str.contains('189_')
    for img_name in rfcnCV[cond].img.unique():
        img = imread('../data/JPEGImagesBlk/%s.jpg'%(img_name))
        bbox = rfcnCV[rfcnCV['img'] == img_name]
        bbox['w'] = bbox['x1'] - bbox['x0']
        bbox['h'] = bbox['y1'] - bbox['y0']
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        for c, row in bbox.iterrows():
            plt.gca().add_patch(plt.Rectangle((row['x0'], row['y0']), row['w'],\
            row['h'], color='red', fill=False, lw=2))

#  
if validate_test:
    ids = [2038]
    for id_ in ids:
        print id_
        cond = rfcnTst.img.str.contains(str(id_)+'_')
        vals = rfcnTst[cond].img.unique()
        for img_name in vals:
            if img_name[:len(str(id_))] == str(id_):
                img = imread('../data/JPEGImagesTest/%s.jpg'%(img_name))
                bbox = rfcnTst[rfcnTst['img'] == img_name]
                bbox['w'] = bbox['x1'] - bbox['x0']
                bbox['h'] = bbox['y1'] - bbox['y0']
                plt.figure(figsize=(7,7))
                plt.imshow(img)
                for c, row in bbox.iterrows():
                    plt.gca().add_patch(plt.Rectangle((row['x0'], row['y0']), row['w'],\
                    row['h'], color='red', fill=False, lw=2))           


if validate_test:
    ids = [163, 306, 322, 886, 923, 1334, 1366, 1411, 1419, 1719, 1810, 1892, 1915, 1976, 1979, 2038]
    ids =[2038]
    for id_ in ids:
        img = imread('../data/Test/%s.jpg'%(id_))
        plt.figure(figsize=(20,20))
        plt.imshow(img)


     
for ii in [2661, 15659,  3656,  2221, 14119, 11852, 17987,  4471, 15707,  2405,  3275,  9843,   163, 13485,  2515, 18081, 14358,  9062, 12384,  8426]:
    img = imread('../data/Test/%s.jpg'%(ii))
    plt.figure(figsize=(10,10))
    plt.imshow(img)

# Ones where we don't have no pups
for ii in [105, 110, 112, 113, 117, 123, 127, 132, 139, 145, 149, 167, 211, 222, 245, 260, 300, 329, 350, 387]:
    img = imread('../data/Train/%s.jpg'%(ii))
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    
for ii in [185, 200, 226 ,244, 251, 267, 275, 275, 288, 288, 289, 296, 305, 322, 486, 592, 682]:
    img = imread('../data/Train/%s.jpg'%(ii))
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    