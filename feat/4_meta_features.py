import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
import xgboost as xgb
import gc

import scipy
from sklearn.metrics import fbeta_score

from PIL import Image
from scipy.ndimage import imread

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

# Load data
os.chdir('/home/darragh/Dropbox/noaa/feat')
columns = ['adult_males','subadult_males','adult_females','juveniles','pups']
block_size = 544
train_path = '../data/Train/'
train_block_path = '../data/JPEGImagesBlk/'
train_dot_path = '../data/TrainDotted/'
test_path = '../data/Test/'
test_block_path = '../data/JPEGImagesTest/'
train = pd.read_csv('../data/train-noaa.csv')
test = pd.read_csv('../data/sample_submission.csv')
train.columns.values[0] = 'id'
test.columns.values[0] = 'id'
big_img_resize = (560,370)
small_img_resize = (544,544)
target_cols = train.columns.values.tolist()
# Load up resnet trn predictions
resn50Tst = pd.read_pickle('../coords/resnet50Preds2604.pkl')
resn50CV = pd.concat([pd.read_pickle('../coords/resnet50CVPreds2604_fold1.pkl'),
                      pd.read_pickle('../coords/resnet50CVPreds2604_fold2.pkl')], axis=0)

def extract_features(df, data_path, (im_size)):
    im_features = df.copy()
    
    mask_size = []    
    
    r_mean = []
    g_mean = []
    b_mean = []

    r_std = []
    g_std = []
    b_std = []

    r_kurtosis = []
    g_kurtosis = []
    b_kurtosis = []
    
    r_skewness = []
    g_skewness = []
    b_skewness = []

    for image_name in tqdm(im_features.id.values, miniters=100):     
        
        im = Image.open(data_path + str(image_name) + '.jpg')               
        im = im.resize(im_size)        
        im = np.array(im)[:,:,:3]     
        
        if data_path == '../data/Train/':
            im_dot = Image.open(data_dot_path + str(image_name) + '.jpg')
            im_dot = im_dot.resize(im_size)
            im_dot = np.array(im_dot)[:,:,:3]  
            im[im_dot==0] = 0
            # Image.fromarray(im, 'RGB').show()
        
        mask_size.append(np.float(np.sum(im[:,:,0].ravel() == 0))/(im_size[0]*im_size[1]))      
        
        ravel_im0 = im[:,:,0].ravel()[im[:,:,0].ravel() != 0]
        ravel_im1 = im[:,:,1].ravel()[im[:,:,1].ravel() != 0]
        ravel_im2 = im[:,:,2].ravel()[im[:,:,2].ravel() != 0]
        
        r_mean.append(np.mean(ravel_im0))
        g_mean.append(np.mean(ravel_im1))
        b_mean.append(np.mean(ravel_im2))

        r_std.append(np.std(ravel_im0))
        g_std.append(np.std(ravel_im1))
        b_std.append(np.std(ravel_im2))

        r_kurtosis.append(scipy.stats.kurtosis(ravel_im0))
        g_kurtosis.append(scipy.stats.kurtosis(ravel_im1))
        b_kurtosis.append(scipy.stats.kurtosis(ravel_im2))
        
        r_skewness.append(scipy.stats.skew(ravel_im0))
        g_skewness.append(scipy.stats.skew(ravel_im1))
        b_skewness.append(scipy.stats.skew(ravel_im2))

    im_features['mask_size'] = mask_size

    im_features['r_mean'] = r_mean
    im_features['g_mean'] = g_mean
    im_features['b_mean'] = b_mean

    im_features['r_std'] = r_std
    im_features['g_std'] = g_std
    im_features['b_std'] = b_std
        
    im_features['r_kurtosis'] = r_kurtosis
    im_features['g_kurtosis'] = g_kurtosis
    im_features['b_kurtosis'] = b_kurtosis
    
    im_features['r_skewness'] = r_skewness
    im_features['g_skewness'] = g_skewness
    im_features['b_skewness'] = b_skewness
    
    return im_features


def extract_block_features(data_path, (im_size)):
    im_features = pd.DataFrame({'id_block': os.listdir(data_path)})
    
    mask_size = []    
    
    r_mean = []
    g_mean = []
    b_mean = []

    r_std = []
    g_std = []
    b_std = []

    r_kurtosis = []
    g_kurtosis = []
    b_kurtosis = []
    
    r_skewness = []
    g_skewness = []
    b_skewness = []

    for image_name in tqdm(im_features.id_block.values, miniters=100):     
        
        im = Image.open(data_path + str(image_name))               
        im = im.resize(im_size)        
        im = np.array(im)[:,:,:3]     
        
        mask_size.append(np.float(np.sum(im[:,:,0].ravel() == 0))/(im_size[0]*im_size[1]))      
        
        ravel_im0 = im[:,:,0].ravel()[im[:,:,0].ravel() != 0]
        ravel_im1 = im[:,:,1].ravel()[im[:,:,1].ravel() != 0]
        ravel_im2 = im[:,:,2].ravel()[im[:,:,2].ravel() != 0]
        
        r_mean.append(np.mean(ravel_im0))
        g_mean.append(np.mean(ravel_im1))
        b_mean.append(np.mean(ravel_im2))

        r_std.append(np.std(ravel_im0))
        g_std.append(np.std(ravel_im1))
        b_std.append(np.std(ravel_im2))

        r_kurtosis.append(scipy.stats.kurtosis(ravel_im0))
        g_kurtosis.append(scipy.stats.kurtosis(ravel_im1))
        b_kurtosis.append(scipy.stats.kurtosis(ravel_im2))
        
        r_skewness.append(scipy.stats.skew(ravel_im0))
        g_skewness.append(scipy.stats.skew(ravel_im1))
        b_skewness.append(scipy.stats.skew(ravel_im2))

    im_features['mask_size'] = mask_size

    im_features['r_mean'] = r_mean
    im_features['g_mean'] = g_mean
    im_features['b_mean'] = b_mean

    im_features['r_std'] = r_std
    im_features['g_std'] = g_std
    im_features['b_std'] = b_std
        
    im_features['r_kurtosis'] = r_kurtosis
    im_features['g_kurtosis'] = g_kurtosis
    im_features['b_kurtosis'] = b_kurtosis
    
    im_features['r_skewness'] = r_skewness
    im_features['g_skewness'] = g_skewness
    im_features['b_skewness'] = b_skewness
    
    return im_features



# Extract meta features
print('Extracting train features')
train_features = extract_features(train, train_path, big_img_resize)
train_features.to_csv('../coords/train_meta1.csv')

print('Extracting test features')
test_features = extract_features(test, test_path, big_img_resize)
test_features.to_csv('../coords/test_meta1.csv')

# Extract block meta features
print('Extracting block train features')
train_block_features = extract_block_features(train_block_path, small_img_resize)
train_block_features.to_csv('../coords/train_block_meta1.csv')

print('Extracting block test features')
test_block_features = extract_block_features(test_block_path, small_img_resize)
test_block_features.to_csv('../coords/test_block_meta1.csv')

# Load up the coords file and aggregate the target to block image level
coords = pd.read_csv("coords_meta.csv")
train_meta = pd.read_csv("train_meta.csv", usecols = ['id', 'height', 'width', 'all_diff'])#,\
train_meta.columns = ['id', 'img_height', 'img_width', 'all_diff']
coords = pd.merge(coords, train_meta, on='id', how='inner')
coords['block_width'] = coords['width'].apply(lambda x: x*img_w).div(coords['img_width'], axis=0).apply(int)%block_size
coords['block_height'] = coords['height'].apply(lambda x: x*img_h).div(coords['img_height'], axis=0).apply(int)%block_size
coords['block'] = coords['width'].apply(lambda x: x*img_w).div(coords['img_width'], axis=0).apply(int).apply(lambda x: x/block_size).apply(str)+\
                    coords['height'].apply(lambda x: x*img_h).div(coords['img_height'], axis=0).apply(int).apply(lambda x: x/block_size).apply(str)
coords['id'] = coords['id'].map(str) + '_' + coords["block"].map(str) + '.jpg'
coords = coords[['id', 'class']]
coords = pd.DataFrame(coords.pivot_table(index='id', columns='class', aggfunc=len, fill_value=0).reset_index())
coords.columns = ['id_block'] + columns

# Merge the train_block_features
train_block_features1 = train_block_features.merge(coords, how='outer', on = 'id_block')
train_block_features1[columns] = train_block_features1[columns].fillna(0)
train_block_features1.to_csv('../coords/train_block_meta1.csv')
for var in columns:
    test_block_features[var] = 0
test_block_features.to_csv('../coords/test_block_meta1.csv')

# Get overlap feature - proportion of overlaps in the predictions
train_features = pd.read_csv('../coords/train_meta1.csv')
test_features  = pd.read_csv('../coords/test_meta1.csv')
train_features.drop(['Unnamed: 0'], axis = 1, inplace = True)
test_features.drop(['Unnamed: 0'], axis = 1, inplace = True)

train_block_features = pd.read_csv('../coords/train_block_meta1.csv')
test_block_features  = pd.read_csv('../coords/test_block_meta1.csv')
train_block_features.drop(['Unnamed: 0'], axis = 1, inplace = True)
test_block_features.drop(['Unnamed: 0'], axis = 1, inplace = True)

resn50CV = resn50CV[resn50CV['predSeal']>0.5].reset_index(drop=True)
resn50Tst = resn50Tst[resn50Tst['predSeal']>0.5].reset_index(drop=True)
resn50CV['bigimg'] = resn50CV['img'].apply(lambda x: int(x.split('_')[0]))
resn50Tst['bigimg'] = resn50Tst['img'].apply(lambda x: int(x.split('_')[0]))
keep_cols = ['img', 'x0', 'y0', 'x1', 'y1']
resn50CV = resn50CV[keep_cols]
resn50Tst = resn50Tst[keep_cols]
gc.collect()

# Function to get area of coverage and overlap of seals
def getOverlap(preddf):
    preddf = preddf.sort('img').reset_index(drop=True)
    imgs = []
    coverage = []
    mat = np.zeros((544, 544))
    for c, row in tqdm(preddf.iterrows(), miniters=100):
        if c==0 : imgprev = row['img']
        if row['img'] != imgprev and c > 0 :
            coverage.append([imgprev, np.sum(mat>0),np.sum(mat>1),np.sum(mat>2)])   
            mat = np.zeros((544, 544))
            imgprev = row['img']
        row_idx = np.array(range(int(row['x0']), int(row['x1'])))   
        col_idx = np.array(range(int(row['y0']), int(row['y1'])))
        mat[row_idx[:, None], col_idx] = mat[row_idx[:, None], col_idx] + 1
    coverage.append([imgprev, np.sum(mat>0),np.sum(mat>1),np.sum(mat>2)]) 
    
    sealCover = pd.DataFrame(coverage, columns = ['img', 'sealCoverage', 'sealOverlap2', 'sealOverlap3'])
    sealCover['bigimg'] = sealCover['img'].apply(lambda x: int(x.split('_')[0]))
    sealCover.drop(['img'], axis=1, inplace=True)
    sealCover = sealCover.groupby(['bigimg']).sum()
    def divide_by_area(x):
        return x.divide(6*9*544*544).astype('float')
    sealCover = sealCover.apply(divide_by_area)
    gc.collect()
    sealOverlapProp2 = []
    sealOverlapProp3 = []
    for c, row in tqdm(sealCover.iterrows(), miniters=100):
        sealOverlapProp2.append(row['sealOverlap2']/row['sealCoverage'])
        sealOverlapProp3.append(row['sealOverlap3']/row['sealCoverage'])
    sealCover['sealOverlapProp2'] = sealOverlapProp2
    sealCover['sealOverlapProp3'] = sealOverlapProp3
    return sealCover    

resn50CVOlap = getOverlap(resn50CV)
resn50TstOlap = getOverlap(resn50Tst)
resn50CVOlap.to_csv('../coords/train_meta2.csv', index=True)
resn50TstOlap.to_csv('../coords/test_meta2.csv')

#resn50CVOlap1 = pd.read_csv('../coords/train_meta2.csv')
#resn50TstOlap1 = pd.read_csv('../coords/test_meta2.csv')
resn50CVOlap.head()

# Function to get area of coverage and overlap of seals of the Blocked images
def getOverlapBlock(preddf):
    preddf = preddf.sort('img').reset_index(drop=True)
    imgs = []
    coverage = []
    mat = np.zeros((544, 544))
    for c, row in tqdm(preddf.iterrows(), miniters=100):
        if c==0 : imgprev = row['img']
        if row['img'] != imgprev and c > 0 :
            coverage.append([imgprev, np.sum(mat>0),np.sum(mat>1),np.sum(mat>2)])   
            mat = np.zeros((544, 544))
            imgprev = row['img']
        row_idx = np.array(range(int(row['x0']), int(row['x1'])))   
        col_idx = np.array(range(int(row['y0']), int(row['y1'])))
        mat[row_idx[:, None], col_idx] = mat[row_idx[:, None], col_idx] + 1
    coverage.append([imgprev, np.sum(mat>0),np.sum(mat>1),np.sum(mat>2)]) 
    
    sealCover = pd.DataFrame(coverage, columns = ['img', 'sealCoverage', 'sealOverlap2', 'sealOverlap3'])
    sealCover = sealCover.groupby(['img']).sum()
    def divide_by_area(x):
        return x.divide(544*544).astype('float')
    sealCover = sealCover.apply(divide_by_area)
    gc.collect()
    sealOverlapProp2 = []
    sealOverlapProp3 = []
    for c, row in tqdm(sealCover.iterrows(), miniters=100):
        sealOverlapProp2.append(row['sealOverlap2']/row['sealCoverage'])
        sealOverlapProp3.append(row['sealOverlap3']/row['sealCoverage'])
    sealCover['sealOverlapProp2'] = sealOverlapProp2
    sealCover['sealOverlapProp3'] = sealOverlapProp3
    return sealCover    

# Calculate the seal coverage and overlap of the blocked images
resn50Tst = pd.read_pickle('../coords/resnet50Preds2604.pkl')
resn50CV = pd.concat([pd.read_pickle('../coords/resnet50CVPreds2604_fold1.pkl'),
                      pd.read_pickle('../coords/resnet50CVPreds2604_fold2.pkl')], axis=0)
resn50CV = resn50CV[resn50CV['predSeal']>0.5].reset_index(drop=True)
resn50Tst = resn50Tst[resn50Tst['predSeal']>0.5].reset_index(drop=True)
keep_cols = ['img', 'x0', 'y0', 'x1', 'y1']
resn50CV = resn50CV[keep_cols]
resn50Tst = resn50Tst[keep_cols]
gc.collect()

resn50CVOlap = getOverlapBlock(resn50CV)
resn50TstOlap = getOverlapBlock(resn50Tst)

resn50CVOlap.to_csv('../coords/train_block_meta2.csv', index=True)
resn50TstOlap.to_csv('../coords/test_block_meta2.csv')
