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
cutoff = 0.3
block_size = 544
img_w = 4896 # width
img_h = 3264 # height
boundaries = [100,80,70,70,40]
ROWS = 100
COLS = 100
check_border = 5
TESTDIR = '../data/Test'
make_train = True
make_test = True
olap_thresh = 0.6

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


df = ssdTst[ssdTst.img.str.contains('2584_')] 

def removeOverlaps(df, threshold):
    tmp = []
    ssdls = []
    df = df.sort(['img', 'proba']).reset_index(drop=True)
    prev_img = df.img[0]

    for c, row in df.iterrows():
        if prev_img == row['img']:
            tmp.append(list(row))
        else:
            if len(tmp)==1:
                ssdls.append(list(tmp[0]))
            else:
                for ii in range(len(tmp)-1):
                    drop = 0
                    img = tmp[ii]
                    img_area = (img[4]-img[2])*(img[5]-img[3])
                    for jj in range(ii+1, len(tmp)):
                        img1 = tmp[jj]
                        intersect = (max(0.0, min(img[4], img1[4]) - max(img[2], img1[2])))*\
                                        (max(0.0, min(img[5], img1[5]) - max(img[3], img1[3])))
                        if intersect > (img_area * threshold) : 
                            drop += 1 
                    if drop == 0 : ssdls.append(list(img))
            tmp = []
        prev_img = row['img']   
    for i in tmp: ssdls.append(list(i))
    return pd.DataFrame(ssdls, columns = df.columns)

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

    def make_CV_boxes(cutoff1, file_name, nms, olap_thresh):
        
        ssdCVfold2 = pd.read_csv("../coords/comp4_det_test_seals_300_"+ nms +"_fold2.txt.zip",\
                            delimiter = " ", header=None, names=['img', 'proba', 'x0', 'y0', 'x1', 'y1'])
        ssdCVfold1 = pd.read_csv("../coords/comp4_det_test_seals_300_"+ nms +"_fold1.txt.zip",\
                            delimiter = " ", header=None, names=['img', 'proba', 'x0', 'y0', 'x1', 'y1'])
        ssdCVfold1['img'] = ssdCVfold1['img'].str.replace('/home/ubuntu/noaa/darknet/seals/JPEGImagesBlk/', '')
        ssdCV = pd.concat([ssdCVfold2, ssdCVfold1])
        ssdCV = ssdCV[ssdCV['proba']>cutoff1]
        ssdCV = ssdCV[(ssdCV['x1']-ssdCV['x0'])<150]
        ssdCV = ssdCV[(ssdCV['y1']-ssdCV['y0'])<150]
        del ssdCVfold1, ssdCVfold2
        gc.collect()
        ssdCV['seal'] = ssdCV.progress_apply(is_seal, axis=1)
        ssdCV = ssdCV.reset_index(drop=True)
        ssdCV['h_diff'] = ROWS - (ssdCV['y1'] -ssdCV['y0'])
        ssdCV['w_diff'] = COLS - (ssdCV['x1'] -ssdCV['x0'])
        ssdCV[ssdCV['h_diff']<0]['h_diff'] = 0
        ssdCV[ssdCV['w_diff']<0]['w_diff'] = 0
        ssdCV['x0'] = ssdCV['x0'] - ssdCV['w_diff'].divide(2)
        ssdCV['x1'] = ssdCV['x1'] + ssdCV['w_diff'].divide(2)
        ssdCV['y0'] = ssdCV['y0'] - ssdCV['h_diff'].divide(2)
        ssdCV['y1'] = ssdCV['y1'] + ssdCV['h_diff'].divide(2)
        ssdCV[['x0', 'x1']] = ssdCV[['x0', 'x1']].add(np.where(ssdCV['x0']<0, ssdCV['x0'].abs(), 0), axis = 0 )
        ssdCV[['y0', 'y1']] = ssdCV[['y0', 'y1']].add(np.where(ssdCV['y0']<0, ssdCV['y0'].abs(), 0), axis = 0 )
        ssdCV[['x0', 'x1']] = ssdCV[['x0', 'x1']].subtract(np.where(ssdCV['x1']>block_size, (ssdCV['x1']-block_size).abs(), 0), axis = 0 )
        ssdCV[['y0', 'y1']] = ssdCV[['y0', 'y1']].subtract(np.where(ssdCV['y1']>block_size, (ssdCV['y1']-block_size).abs(), 0), axis = 0 )
        ssdCV.drop(['h_diff', 'w_diff'], axis=1, inplace=True)
        #ssdCV.to_pickle('../coords/ssdCV.pkl')
        ssdCV = removeOverlaps(ssdCV, olap_thresh)
        ssdCV.to_pickle(file_name)
        return ssdCV
    ssdCV = make_CV_boxes(cutoff, '../coords/ssdCV_nms0.45.pkl', 'nms0.45')
    ssdCV = make_CV_boxes(cutoff, '../coords/ssdCV_nms0.75.pkl', 'nms0.75')
else:
    ssdCV = pd.read_pickle('../coords/ssdCV_nms0.45.pkl' )
    ssdCV = pd.read_pickle('../coords/ssdCV_nms0.75.pkl' )

# Cancel overlapping images
ssdCV.seal.hist()




        
ssdCV1 = removeOverlaps(ssdCV, 0.8)

# Load the Xtrain files
if make_test:
    def make_Tst_boxes(cutoff1, file_name, nms):
        ssdTst = pd.read_csv("../coords/comp4_det_test_seals_300_"+ nms +"_all.txt.zip", skiprows=1, \
                            delimiter = " ", header=None, names=['img', 'proba', 'x0', 'y0', 'x1', 'y1'])
        ssdTst['img'] = ssdTst['img'].str.replace('/home/ubuntu/noaa/darknet/seals/JPEGImagesTest/', '')
        ssdTst = ssdTst[ssdTst['proba']>cutoff1]
        ssdTst = ssdTst[(ssdTst['x1']-ssdTst['x0'])<150]
        ssdTst = ssdTst[(ssdTst['y1']-ssdTst['y0'])<150]
        gc.collect()
        ssdTst = ssdTst.reset_index(drop=True)
        ssdTst['h_diff'] = ROWS - (ssdTst['y1'] -ssdTst['y0'])
        ssdTst['w_diff'] = COLS - (ssdTst['x1'] -ssdTst['x0'])
        ssdTst[ssdTst['h_diff']<0]['h_diff'] = 0
        ssdTst[ssdTst['w_diff']<0]['w_diff'] = 0
        ssdTst['x0'] = ssdTst['x0'] - ssdTst['w_diff'].divide(2)
        ssdTst['x1'] = ssdTst['x1'] + ssdTst['w_diff'].divide(2)
        ssdTst['y0'] = ssdTst['y0'] - ssdTst['h_diff'].divide(2)
        ssdTst['y1'] = ssdTst['y1'] + ssdTst['h_diff'].divide(2)
        ssdTst[['x0', 'x1']] = ssdTst[['x0', 'x1']].add(np.where(ssdTst['x0']<0, ssdTst['x0'].abs(), 0), axis = 0 )
        ssdTst[['y0', 'y1']] = ssdTst[['y0', 'y1']].add(np.where(ssdTst['y0']<0, ssdTst['y0'].abs(), 0), axis = 0 )
        ssdTst[['x0', 'x1']] = ssdTst[['x0', 'x1']].subtract(np.where(ssdTst['x1']>block_size, (ssdTst['x1']-block_size).abs(), 0), axis = 0 )
        ssdTst[['y0', 'y1']] = ssdTst[['y0', 'y1']].subtract(np.where(ssdTst['y1']>block_size, (ssdTst['y1']-block_size).abs(), 0), axis = 0 )
        ssdTst.drop(['h_diff', 'w_diff'], axis=1, inplace=True)        
        ssdTst = removeOverlaps(ssdTst, olap_thresh)
        ssdTst.to_pickle(file_name)
        return ssdTst
    ssdTst = make_Tst_boxes(cutoff, '../coords/ssdTst_nms0.45.pkl', 'nms0.45')
    ssdTst = make_Tst_boxes(cutoff, '../coords/ssdTst_nms0.75.pkl', 'nms0.75')
else:
    ssdTst = pd.read_pickle('../coords/ssdTst_nms0.45.pkl')
    ssdTst = pd.read_pickle('../coords/ssdTst_nms0.75.pkl')
 


# Lets validate the train file
if validate_test:
    cond = ssdCV.img.str.contains('189_')
    for img_name in ssdCV[cond].img.unique():
        img = imread('../data/JPEGImagesBlk/%s.jpg'%(img_name))
        bbox = ssdCV[ssdCV['img'] == img_name]
        bbox['w'] = bbox['x1'] - bbox['x0']
        bbox['h'] = bbox['y1'] - bbox['y0']
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        for c, row in bbox.iterrows():
            plt.gca().add_patch(plt.Rectangle((row['x0'], row['y0']), row['w'],\
            row['h'], color='red', fill=False, lw=2))

#  
if validate_test:
    ids = ['163_']
    for id_ in ids:
        print id_
        cond = ssdTst.img.str.contains(str(id_))
        vals = ssdTst[cond].img.unique()
        for img_name in vals:
            if img_name[:len(str(id_))] == str(id_):
                img = imread('../data/JPEGImagesTest/%s.jpg'%(img_name))
                bbox = ssdTst[ssdTst['img'] == img_name]
                bbox['w'] = bbox['x1'] - bbox['x0']
                bbox['h'] = bbox['y1'] - bbox['y0']
                plt.figure(figsize=(7,7))
                plt.imshow(img)
                for c, row in bbox.iterrows():
                    plt.gca().add_patch(plt.Rectangle((row['x0'], row['y0']), row['w'],\
                    row['h'], color='red', fill=False, lw=2))           


if validate_test:
    ids = [163]#, 306, 322, 886, 923, 1334]#, 1366, 1411, 1419, 1719, 1810, 1892, 1915, 1976, 1979, 2038]
    for id_ in ids:
        img = imread('../data/Test/%s.jpg'%(id_))
        plt.figure(figsize=(10,10))
        plt.imshow(img)


     
for ii in [2661, 15659,  3656,  2221, 14119, 11852, 17987,  4471, 15707,  2405,  3275,  9843,   163, 13485,  2515, 18081, 14358,  9062, 12384,  8426]:
    img = imread('../data/Test/%s.jpg'%(ii))
    plt.figure(figsize=(10,10))
    plt.imshow(img)

# Ones where we don't have no pups
for ii in [105]:#, 110, 112, 113, 117, 123, 127, 132, 139, 145, 149, 167, 211, 222, 245, 260, 300, 329, 350, 387]:
    img = imread('../data/Train/%s.jpg'%(ii))
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    
for ii in [185, 200, 226 ,244, 251, 267, 275, 275, 288, 288, 289, 296, 305, 322, 486, 592, 682]:
    img = imread('../data/Train/%s.jpg'%(ii))
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    