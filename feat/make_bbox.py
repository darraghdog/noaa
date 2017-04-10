# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:27:44 2017

@author: darragh
"""


import os
import pandas as pd
from matplotlib import pyplot as plt
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from scipy.misc import imsave

# 'adult_males', 'subadult_males','adult_females','juveniles','pups',
os.chdir('/home/darragh/Dropbox/noaa/feat')
boundaries = [100,80,70,70,40]
colors = ['red', 'blue', 'green', 'yellow', 'pink']
validate = False
bbimg_trn = False
boundary = 40
block_size = 544
img_w = 4896 # width
img_h = 3264 # height

def create_rect(row):
    h = row['height']
    w = row['width']
    c = row['class']
    col = colors[c]
    b = boundaries[c] #boundary
    return plt.Rectangle((h-b/2, w-b/2), b, b, color=col, fill=False, lw=2)

def create_rect2(row):
    h = row['y1'] - row['y0']
    w = row['x1'] - row['x0']
    c = row['class']
    return plt.Rectangle((row['x0'], row['y0']), w, h, color=row['colors'], fill=False, lw=2)

# Load up the coords file
coords = pd.read_csv("coords_meta.csv")
train_meta = pd.read_csv("train_meta.csv", usecols = ['id', 'height', 'width', 'all_diff'])#,\
train_meta.columns = ['id', 'img_height', 'img_width', 'all_diff']
coords = pd.merge(coords, train_meta, on='id', how='inner')

# Test on an image
if validate:
    samp = 20 # try one sample image
    samp_coords = coords[coords['id']==samp]
    img = imread('../data/Train/%s.jpg'%(samp))
    
    plt.figure(figsize=(50,80))
    plt.imshow(img)
    for c, row in samp_coords.iterrows():
        if row['class'] < 5:
            plt.gca().add_patch(create_rect(row))

# Split the coords into 540 blocks
# Resize images to 4896 (544*9) * 3264 (544*6)
# Break the coords into blocks of 544 after resizing each image to 4896*3264
coords['block_width'] = coords['width'].apply(lambda x: x*img_w).div(coords['img_width'], axis=0).apply(int)%block_size
coords['block_height'] = coords['height'].apply(lambda x: x*img_h).div(coords['img_height'], axis=0).apply(int)%block_size
coords['block'] = coords['width'].apply(lambda x: x*img_w).div(coords['img_width'], axis=0).apply(int).apply(lambda x: x/block_size).apply(str)+\
                    coords['height'].apply(lambda x: x*img_h).div(coords['img_height'], axis=0).apply(int).apply(lambda x: x/block_size).apply(str)
coords.head(3)

# get the coords boxes
classbox = pd.DataFrame.from_items([('class', range(len(colors))),
                         ('colors', colors),
                         ('boundaries', boundaries)])
coords = pd.merge(coords, classbox, on='class', how='inner')
coords['x0'] = (coords['block_width']-coords['boundaries'].div(2)).clip(lower=0, upper=544)
coords['y0'] = (coords['block_height']-coords['boundaries'].div(2)).clip(lower=0, upper=544)
coords['x1'] = (coords['block_width']+coords['boundaries'].div(2)).clip(lower=0, upper=544)
coords['y1'] = (coords['block_height']+coords['boundaries'].div(2)).clip(lower=0, upper=544)


# Get the unique blocks with seals for each image and start saving them as separate images
# These will be our 544 * 544 images with seals in them
blocks = coords[['id', 'block']].drop_duplicates().reset_index(drop=True)
if bbimg_trn:
    for ii in blocks.id.unique():
        if ii%50==0: print 'Image # '+str(ii)
        idx = blocks['id']==ii
        img = imread('../data/Train/%s.jpg'%(ii))
        img = scipy.misc.imresize(img, (img_h, img_w), interp='nearest')
        for c, rows in blocks[idx].iterrows():
            block = rows['block']
            blkw, blkh = int(block[0])*block_size, int(block[1])*block_size
            img_dump = img[(blkh):(blkh+block_size), (blkw):(blkw+block_size)]
            imsave('../data/JPEGImages/%s_%s.jpg'%(ii, block), img_dump)
        
# Check the resized boundary box works with the resized image
# Test on an image
if validate:
    samp = 100 # try one sample image
    block = blocks.iloc[samp]
    img = imread('../data/JPEGImages/%s_%s.jpg'%(block['id'], block['block']))
    bbox = coords[coords['id']==block['id']][coords['block']==block['block']]
    plt.figure(figsize=(15,15))
    plt.imshow(img)
    for c, row in bbox.iterrows():
        if row['class'] < 4:
            plt.gca().add_patch(create_rect2(row))

# Send this file to disk
coords.to_csv("block_coords.csv", index=False)

# Write out VOC labels
for c, irow in blocks.iterrows():
    bbox = coords[coords['id']==block['id']][coords['block']==block['block']]
    fo = open(os.path.join("yolo_labels/seals", '%s_%s.txt'%(irow['id'], irow['block'])),'w')
    for c, row in bbox.iterrows():
        dimvals = [row['x0'], row['y0'], row['x1'] - row['x0'], row['y1'] - row['y0']]
        if row['class'] < 4:
            fo.write('%s %s\n'%('0',' '.join(map(str, map(int, dimvals)))))
    del bbox
    fo.close()	
