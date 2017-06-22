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

# 'adult_males', 'subadult_males','adult_females','juveniles','pups',
os.chdir('/home/darragh/Dropbox/noaa/feat')
boundaries = [80,70,70,70,50]
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
MASK_FOLDER = '/home/darragh/Dropbox/noaa/data/mask/classes'

def makeDir(name):
    if not os.path.exists(name):
        os.mkdir(name)

# Load up the coordinates
coords = pd.read_csv("../coords/block_coords.csv")
coords = coords[['id', 'block', 'class', 'block_width', 'block_height']]
img_files = coords[['id', 'block']].drop_duplicates().reset_index(drop=True)

# Process training images
vggCVpred = pd.concat([pd.read_csv('../coords/vggCVPreds2604_fold1.csv'),
           pd.read_csv('../coords/vggCVPreds2604_fold2.csv')], axis = 0)
idx = vggCVpred.groupby(['img'])['predSeal'].transform(max) == vggCVpred['predSeal']
vggCVpred = vggCVpred[idx]
vggCVpred = vggCVpred[vggCVpred['predSeal']>0.6].reset_index(drop=True)
vggCVpred['id'] = vggCVpred['img'].map(lambda x: int(x.split('_')[0]))
vggCVpred['block'] = vggCVpred['img'].map(lambda x: x.split('_')[1])

img_all = os.listdir('../data/JPEGImagesBlk')
for c, row in vggCVpred.reset_index(drop=True).iterrows():
    tif = np.zeros((544,544, 6), dtype=np.uint8)
    if row['id'] in coords.id:
        dftmp = coords[(coords['id']==row['id']) & (coords['block']==int(row['block'])) ].reset_index(drop=True)
        for c1, row1 in dftmp.iterrows():
            radius = boundaries[row1['class']]/2
            rr, cc = draw.circle(row1['block_height'], row1['block_width'], radius=radius, shape=tif.shape)
            tif[rr, cc, row1['class']] = 1
    
    np.save(MASK_FOLDER +'/%s_%s'%(row[2], row[3]), tif)