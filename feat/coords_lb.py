"""Sea Lion Prognostication Engine

https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count
"""

import sys
import os
from collections import namedtuple
import operator
import glob
import csv 
from math import sqrt
import cPickle
from operator import sub

import numpy as np

import PIL
from PIL import Image, ImageDraw, ImageFilter

import skimage
import skimage.io
import skimage.measure

import shapely
import shapely.geometry
from shapely.geometry import Polygon

# Notes
# cls -- sea lion class 
# tid -- train, train dotted, or test image id 
# _nb -- short for number
# x, y -- don't forget image arrays organized row, col, channels
#
# With contributions from @bitsofbits ...
#


# ================ Meta ====================
__description__ = 'Sea Lion Prognostication Engine'
__version__ = '0.1.0'
__license__ = 'MIT'
__author__ = 'Gavin Crooks (@threeplusone)'
__status__ = "Prototype"
__copyright__ = "Copyright 2017"

# python -c 'import sealiondata; sealiondata.package_versions()'
def package_versions():
    print('sealionengine \t', __version__)
    print('python        \t', sys.version[0:5])
    print('numpy         \t', np.__version__)
    print('skimage       \t', skimage.__version__)
    print('pillow (PIL)  \t', PIL.__version__)
    print('shapely       \t', shapely.__version__)


SOURCEDIR = os.path.join('..', 'data')

DATADIR = '.'

VERBOSITY = namedtuple('VERBOSITY', ['QUITE', 'NORMAL', 'VERBOSE', 'DEBUG'])(0,1,2,3)


SeaLionCoord = namedtuple('SeaLionCoord', ['tid', 'cls', 'x', 'y'])

allfiles = os.listdir(SOURCEDIR+'/Train')


class SeaLionData(object):
    
    def __init__(self, sourcedir=SOURCEDIR, datadir=DATADIR, verbosity=VERBOSITY.NORMAL):
        self.sourcedir = sourcedir
        self.datadir = datadir
        self.verbosity = verbosity
        
        self.cls_nb = 5
        
        self.cls_names = (
            'adult_males',
            'subadult_males',
            'adult_females',
            'juveniles',
            'pups',
            'NOT_A_SEA_LION')
            
        self.cls = namedtuple('ClassIndex', self.cls_names)(*range(0,6))
    
        # backported from @bitsofbits. Average actual color of dot centers.
        self.cls_colors = (
            (243,8,5),          # red
            (244,8,242),        # magenta
            (87,46,10),         # brown 
            (25,56,176),        # blue
            (38,174,21),        # green
            )
    
            
        self.dot_radius = 3
        
        self.train_nb = 947
        
        self.test_nb = 18636
       
        self.paths = {
            # Source paths
            'sample'     : os.path.join(sourcedir, 'sample_submission.csv'),
            'counts'     : os.path.join(sourcedir, 'Train', 'train.csv'),
            'train'      : os.path.join(sourcedir, 'Train', '{tid}.jpg'),
            'dotted'     : os.path.join(sourcedir, 'TrainDotted', '{tid}.jpg'),
            'test'       : os.path.join(sourcedir, 'Test', '{tid}.jpg'),
            # Data paths
            'coords'     : os.path.join(datadir, 'coords.csv'),  
            }
        
        # From MismatchedTrainImages.txt
        self.bad_train_ids = (
            3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 
            268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 
            507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 
            779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 
            913, 927, 946)
            
        self._counts = None

        
    @property
    def trainshort_ids(self):
        return (1,2, 4)#,0,5,6,8,10)  # Trainshort1
        #return range(41,51)         # Trainshort2
        
    @property 
    def train_ids(self):
        """List of all valid train ids"""
        tids = range(0, self.train_nb)
        tids = list(set(tids) - set(self.bad_train_ids) )  # Remove bad ids
        tids.sort()
        return tids
                    
    @property 
    def test_ids(self):
        return range(0, self.test_nb)
    
    def path(self, name, **kwargs):
        """Return path to various source files"""
        path = self.paths[name].format(**kwargs)
        return path        

    @property
    def counts(self) :
        """A map from train_id to list of sea lion class counts"""
        if self._counts is None :
            counts = {}
            fn = self.path('counts')
            with open(fn) as f:
                f.readline()
                for line in f:
                    tid_counts = list(map(int, line.split(',')))
                    counts[tid_counts[0]] = tid_counts[1:]
            self._counts = counts
        return self._counts

    def rmse(self, tid_counts) :
        true_counts = self.counts
        
        error = np.zeros(shape=[5] )
        
        for tid in tid_counts:
            true_counts = self.counts[tid]
            obs_counts = tid_counts[tid]
            diff = np.asarray(true_counts) - np.asarray(obs_counts)
            error += diff*diff
        #print(error)
        error /= len(tid_counts)
        rmse = np.sqrt(error).sum() / 5
        return rmse 
        

    def load_train_image(self, train_id, border=0, mask=False):
        """Return image as numpy array
         
        border -- add a black border of this width around image
        mask -- If true mask out masked areas from corresponding dotted image
        """
        img = self._load_image('train', train_id, border)
        if mask :
            # The masked areas are not uniformly black, presumable due to 
            # jpeg compression artifacts
            dot_img = self._load_image('dotted', train_id, border).astype(np.uint16).sum(axis=-1)
            img = np.copy(img)
            img[dot_img<40] = 0
        return img
   

    def load_dotted_image(self, train_id, border=0):
        return self._load_image('dotted', train_id, border)
 
 
    def load_test_image(self, test_id, border=0):    
        return self._load_image('test', test_id, border)


    def _load_image(self, itype, tid, border=0) :
        fn = self.path(itype, tid=tid)
        img = np.asarray(Image.open(fn))
        if border :
            height, width, channels = img.shape
            bimg = np.zeros( shape=(height+border*2, width+border*2, channels), dtype=np.uint8)
            bimg[border:-border, border:-border, :] = img
            img = bimg
        return img
    

    def coords(self, train_id):
        """Extract coordinates of dotted sealions and return list of SeaLionCoord objects)"""
        
        # Empirical constants
        MIN_DIFFERENCE = 16
        MIN_AREA = 9
        MAX_AREA = 100
        MAX_AVG_DIFF = 50
        MAX_COLOR_DIFF = 32
       
        src_img = np.asarray(self.load_train_image(train_id, mask=True), dtype = np.float)
        dot_img = np.asarray(self.load_dotted_image(train_id), dtype = np.float)

        img_diff = np.abs(src_img-dot_img)
        
        # Detect bad data. If train and dotted images are very different then somethings wrong.
        avg_diff = img_diff.sum() / (img_diff.shape[0] * img_diff.shape[1])
        if avg_diff > MAX_AVG_DIFF: return None
        
        img_diff = np.max(img_diff, axis=-1)   
           
        img_diff[img_diff<MIN_DIFFERENCE] = 0
        img_diff[img_diff>=MIN_DIFFERENCE] = 255

        sealions = []
        img_meta = [train_id, src_img.shape[0], src_img.shape[1]]
        
        for cls, color in enumerate(self.cls_colors):
            # color search backported from @bitsofbits.
            color_array = np.array(color)[None, None, :]
            has_color = np.sqrt(np.sum(np.square(dot_img * (img_diff > 0)[:,:,None] - color_array), axis=-1)) < MAX_COLOR_DIFF 
            contours = skimage.measure.find_contours(has_color.astype(float), 0.5)
            
            if self.verbosity == VERBOSITY.DEBUG :
                print()
                fn = 'diff_{}_{}.png'.format(train_id,cls)
                print('Saving train/dotted difference: {}'.format(fn))
                Image.fromarray((has_color*255).astype(np.uint8)).save(fn)

            for cnt in contours :
                p = Polygon(shell=cnt)
                area = p.area 
                if(area > MIN_AREA and area < MAX_AREA) :
                    y, x= p.centroid.coords[0] # DANGER : skimage and cv2 coordinates transposed?
                    x = int(round(x))
                    y = int(round(y))
                    sealions.append( SeaLionCoord(train_id, cls, x, y) )
                    
                
        if self.verbosity >= VERBOSITY.VERBOSE :
            counts = [0,0,0,0,0]
            for c in sealions :
                counts[c.cls] +=1
            print()
            print('train_id','true_counts','counted_dots', 'difference')#, sep='\t')   
            true_counts = self.counts[train_id]
            print(train_id, true_counts, counts, np.array(true_counts) - np.array(counts))# , sep='\t' )
            img_meta.append(true_counts)
            img_meta.append(counts)
          
        if self.verbosity == VERBOSITY.DEBUG :
            img = np.copy(sld.load_dotted_image(train_id))
            r = self.dot_radius
            dy,dx,c = img.shape
            for tid, cls, cx, cy in sealions :                    
                for x in range(cx-r, cx+r+1) : img[cy, x, :] = 255
                for y in range(cy-r, cy+r+1) : img[y, cx, :] = 255    
            fn = 'cross_{}.png'.format(train_id)
            print('Saving crossed dots: {}'.format(fn))
            Image.fromarray(img).save(fn)
     
        return sealions, img_meta
        

    def save_coords(self, train_ids=None): 
        if train_ids is None: train_ids = self.train_ids
        fn = self.path('coords')
        self._progress('Saving sealion coordinates to {}'.format(fn))
        with open(fn, 'w') as csvfile:
            writer =csv.writer(csvfile)
            writer.writerow( SeaLionCoord._fields )
            for tid in train_ids :
                self._progress()
                for coord in self.coords(tid):
                    writer.writerow(coord)
        self._progress('done')
        
    def load_coords(self):
        fn = self.path('coords')
        self._progress('Loading sea lion coordinates from {}'.format(fn))
        with open(fn) as f:
            f.readline()
            return [SeaLionCoord(*[int(n) for n in line.split(',')]) for line in f]

    
            
    def save_sea_lion_chunks(self, coords, chunksize=128):
        self._progress('Saving image chunks...')
        self._progress('\n', verbosity=VERBOSITY.VERBOSE)
        
        last_tid = -1
        
        for tid, cls, x, y in coords :
            if tid != last_tid:
                img = self.load_train_image(tid, border=chunksize//2, mask=True)
                last_tid = tid

            fn = 'chunk_{tid}_{cls}_{x}_{y}_{size}.png'.format(size=chunksize, tid=tid, cls=cls, x=x, y=y)
            self._progress(' Saving '+fn, end='\n', verbosity=VERBOSITY.VERBOSE)
            Image.fromarray( img[y:y+chunksize, x:x+chunksize, :]).save(fn)
            self._progress()
        self._progress('done')
        
            
    def _progress(self, string=None, end=' ', verbosity=VERBOSITY.NORMAL):
        if self.verbosity < verbosity: return
        if not string :
            print('.')#, end='')
        elif string == 'done':
            print(' done') 
        else:
            print(string)#, end=end)
        sys.stdout.flush()

# end SeaLionData


# Count sea lion dots and compare to truth from train.csv
sld = SeaLionData()
sld.verbosity = VERBOSITY.VERBOSE
meta_dict = {}
coords = []
#for tid in sld.trainshort_ids:
for tid in sld.train_ids:
    coord, img_meta = sld.coords(tid)
    coords += coord
    meta_dict[img_meta[0]] = img_meta[1:]

with open(r"coords.pickle", "wb") as output_file:
    cPickle.dump(coords, output_file)
with open(r"meta.pickle", "wb") as output_file:
    cPickle.dump(meta_dict, output_file)   
    
with open(r"coords.pickle", "rb") as input_file:
    coords_in = cPickle.load(input_file)
with open(r"meta.pickle", "rb") as input_file:
    meta_in = cPickle.load(input_file)

# Write out train meta file
fo = open('train_meta.csv','w')
fo.write('id,height,width,all_diff,total_act,adult_males_dots,subadult_males_dots,adult_females_dots,juveniles_dots,pups_dots,\
                 adult_males_act,subadult_males_act,adult_females_act,juveniles_act,pups_act\n')
for i in meta_in:
    vals = meta_in[i]    
    diff = sum(map(abs, map(sub, vals[2], vals[3])))
    tot_act = sum(vals[3])
    vals = vals[:2]+[diff, tot_act]+vals[2]+vals[3]
    fo.write('%s,%s\n'%(i,','.join(map(str, vals))))
    del vals, i
fo.close()	


# Write out train meta file
fo = open('coords_meta.csv','w')
fo.write('id,class,width,height\n')
for i in coords_in:
    fo.write('%s\n'%(','.join(map(str, i))))
    del i
fo.close()	
