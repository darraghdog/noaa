import os
import math
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
# os.chdir('/home/darragh/Dropbox/noaa/feat')
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
img_rows = 544
img_cols = 544
img_w = 4896 # width
img_h = 3264 # height
classes = 5
MASK_FOLDER = '/home/ubuntu/noaa/data/mask/classes/train'

def makeDir(name):
    if not os.path.exists(name):
        os.mkdir(name)


def show_mask(img):
    imout = np.ndarray((img.shape[0]*2, img.shape[1]*3), dtype=np.uint8)
    for i in range(6):
        y_pos, x_pos = math.floor(i/3)*img.shape[0], i%3*img.shape[0]
        imout[int(y_pos):int((y_pos+img.shape[0])), int(x_pos):int((x_pos+img.shape[0]))] = img[:,:,i]
        imout[int(y_pos):int(y_pos)+2,:] = 1
        imout[:,int(x_pos):int(x_pos)+2] = 1
    plt.imshow(imout)
    plt.show()
    
def multi_resize(img_mask, image_rows=img_rows, image_cols=img_cols, classes=classes):
    imout = np.ndarray((image_rows, image_cols, classes), dtype=np.uint8)
    for i in range(classes):
        imout[:,:,i] = resize(img_mask[:,:,i].astype(np.float32), (img_rows, img_cols), mode='reflect')
    return imout
    
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

img_all = os.listdir('../darknet/seals/JPEGImagesBlk')
for c, row in vggCVpred.reset_index(drop=True).iterrows():
    if c % 1000== 0 : 
        print "Row: " + str(c)
    tif = np.zeros((544,544, classes), dtype=np.uint8)
    if row['id'] in coords.id:
        dftmp = coords[(coords['id']==row['id']) & (coords['block']==int(row['block'])) ].reset_index(drop=True)
        for c1, row1 in dftmp.iterrows():
            radius = boundaries[row1['class']]/2
            rr, cc = draw.circle(row1['block_height'], row1['block_width'], radius=radius, shape=tif.shape)
            tif[rr, cc, row1['class']] = 1
    
    np.save(MASK_FOLDER +'/%s_%s'%(row[2], row[3]), tif)
    

 #show_mask(tif)