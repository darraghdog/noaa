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
from PIL import Image

# 'adult_males', 'subadult_males','adult_females','juveniles','pups',
os.chdir('/home/darragh/Dropbox/noaa/feat')
boundaries = [100,80,70,70,40]
colors = ['red', 'blue', 'green', 'yellow', 'pink']
validate = False
bbimg_trn = False
bbimgblk_trn = False
annotations = False
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

def create_rect3(row):
    c = row['class']
    return plt.Rectangle((row['x0']*544, row['y0']*544), row['w']*544, row['h']*544, color='red', fill=False, lw=2)

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

if bbimgblk_trn:
    for ii in blocks.id.unique():
        if ii > 1:break
        if ii%50==0: print 'Image # '+str(ii)
        idx = blocks['id']==ii
        img_in = imread('../data/Train/%s.jpg'%(ii))
        img_blk = imread('../data/TrainDotted/%s.jpg'%(ii))
        img = img_in
        img[img_blk==0]=0
        img = scipy.misc.imresize(img, (img_h, img_w), interp='nearest')
        for c, rows in blocks[idx].iterrows():
            block = rows['block']
            blkw, blkh = int(block[0])*block_size, int(block[1])*block_size
            img_dump = img[(blkh):(blkh+block_size), (blkw):(blkw+block_size)]
            imsave('../data/JPEGImagesBlk/%s_%s.jpg'%(ii, block), img_dump)

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
# coords = pd.read_csv("block_coords.csv")

def decsc(a):
    return '{:.7f}'.format(a)

# Write out VOC labels
for c, irow in blocks.iterrows():
    bbox = coords[coords['id']==irow['id']][coords['block']==irow['block']]
    fo = open(os.path.join("../data/yolo_labels/seals", '%s_%s.txt'%(irow['id'], irow['block'])),'w')
    for c, row in bbox.iterrows():
        dimvals = [(row['x0']/544.0)+.00001, (row['y0']/544.0)+.00001, \
            ((row['x1'] - row['x0'])/544.0)-.00001, ((row['y1'] - row['y0'])/544.0)-.00001]
        if row['class'] < 4:
            fo.write('%s %s\n'%('0',' '.join(map(decsc, dimvals))))
        
    del bbox
    fo.close()	

# Write out image file locations
image_ids = []
base_dir = os.path.join('/home/darragh/Dropbox/noaa', 'data/JPEGImages/')
base_dir_ubuntu = '/home/ubuntu/noaa/darknet/seals/JPEGImages/'
f = os.listdir(base_dir)

# Train test split - evens in train; odds are test
ftrn = [base_dir_ubuntu + s for s in f if int(s.split('_')[0])%2 == 0]
ftst = [base_dir_ubuntu + s for s in f if int(s.split('_')[0])%2 == 1]

# Write out yolo train file
list_file = open('../data/yolo_labels/train.txt', 'w')
for fl in ftrn:
    list_file.write(fl + '\n')
list_file.close()

# Write out yolo test file
list_file = open('../data/yolo_labels/test.txt', 'w')
for fl in ftst:
    list_file.write(fl + '\n')
list_file.close()

# Verify yolo files
if validate:
    samp = 100 # try one sample image
    block = blocks.iloc[samp]
    img = imread('../data/JPEGImages/%s_%s.jpg'%(block['id'], block['block']))
    bbox = pd.read_csv('../data/yolo_labels/seals/%s_%s.txt'%(block['id'], block['block']),\
                    delimiter = " ", header=None, names=['class', 'x0', 'y0', 'w', 'h'])
    plt.figure(figsize=(15,15))
    plt.imshow(img)
    
    for c, row in bbox.iterrows():
        if row['class'] < 4:
            plt.gca().add_patch(create_rect3(row))

# #write Annotations
import glob
if not os.path.exists('../data/Annotations'):
    os.mkdir('../data/Annotations')
files = glob.glob('../data/Annotations/*')
for f in files:
    os.remove(f)

# Write out annotation files for RFCN
if annotations:
    c= "seals" # Only count seals for a start
    TRAIN_DIR = "../data/JPEGImagesBlk"
    for n, block in blocks.iterrows():
        # if n>102:break
        bbox = pd.read_csv('../data/yolo_labels/seals/%s_%s.txt'%(block['id'], block['block']),\
                     delimiter = " ", header=None, names=['class', 'x0', 'y0', 'w', 'h'])
        filename = '%s_%s.jpg'%(block['id'], block['block'])
        tail = filename
        basename, file_extension = os.path.splitext(tail) 
        if len(bbox) == 0:
            print(filename)
            print("no bbox")
        else:
            f = open('../data/Annotations/' + basename + '.xml','w') 
            line = "<annotation>" + '\n'
            f.write(line)
            line = '\t<folder>' + c + '</folder>' + '\n'
            f.write(line)
            line = '\t<filename>' + tail + '</filename>' + '\n'
            f.write(line)
            line = '\t<source>\n\t\t<database>Source</database>\n\t</source>\n'
            f.write(line)
            # im=Image.open(TRAIN_DIR+ c + '/' + tail)
            im=Image.open(TRAIN_DIR + '/' + tail)
            (width, height) = im.size
            line = '\t<size>\n\t\t<width>'+ str(width) + '</width>\n\t\t<height>' + \
            str(height) + '</height>\n\t\t<depth>3</depth>\n\t</size>'
            f.write(line)
            line = '\n\t<segmented>0</segmented>'
            f.write(line)
            for a in bbox.iterrows():
                a = list(a[1])
                line = '\n\t<object>'
                #line += '\n\t\t<name>' + a["class"].lower() + '</name>\n\t\t<pose>Unspecified</pose>'
                line += '\n\t\t<name>' + c + '</name>\n\t\t<pose>Unspecified</pose>'
                #line += '\n\t\t<name>fish</name>\n\t\t<pose>Unspecified</pose>'
                line += '\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>'
                xmin = (round(a[1]*544,1))
                line += '\n\t\t<bndbox>\n\t\t\t<xmin>' + str(xmin) + '</xmin>'
                ymin = (round(a[2]*544,1))
                line += '\n\t\t\t<ymin>' + str(ymin) + '</ymin>'
                width = (round(a[3]*544,1))
                height = (round(a[4]*544,1))
                xmax = xmin + width
                ymax = ymin + height
                line += '\n\t\t\t<xmax>' + str(xmax) + '</xmax>'
                line += '\n\t\t\t<ymax>' + str(ymax) + '</ymax>'
                line += '\n\t\t</bndbox>'
                line += '\n\t</object>'     
                f.write(line)
            line = '</annotation>'
            f.write(line)
            f.close()

#write train ImageSets/Main
if not os.path.exists('../data/ImageSets'):
    os.mkdir('../data/ImageSets')
if not os.path.exists('../data/ImageSets/Main'):
    os.mkdir('../data/ImageSets/Main')
files = glob.glob('../data/ImageSets/Main/*')
for f in files:
    os.remove(f)
    
trn_img = [str(row[1][0])+'_'+str(row[1][1]) for row in blocks.iterrows() if row[1][0]%2==0]
tst_img = [str(row[1][0])+'_'+str(row[1][1]) for row in blocks.iterrows() if row[1][0]%2==1]

with open('../data/ImageSets/Main/trainval.txt','w') as f:
    for im in trn_img:
        f.write(im + '\n')

with open('../data/ImageSets/Main/test.txt','w') as f:
    for im in tst_img:
        f.write(im + '\n')

