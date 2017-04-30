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
train_path = '../data/Train/'
train_dot_path = '../data/TrainDotted/'
test_path = '../data/Test/'
train = pd.read_csv('../data/train-noaa.csv')
test = pd.read_csv('../data/sample_submission.csv')
train.columns.values[0] = 'id'
test.columns.values[0] = 'id'
big_img_resize = (560,370)
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

# Extract meta features
print('Extracting train features')
train_features = extract_features(train, train_path, big_img_resize)
train_features.to_csv('../coords/train_meta1.csv')

print('Extracting test features')
test_features = extract_features(test, test_path, big_img_resize)
test_features.to_csv('../coords/test_meta1.csv')

# Get overlap feature - proportion of overlaps in the predictions
train_features = pd.read_csv('../coords/train_meta1.csv')
test_features  = pd.read_csv('../coords/test_meta1.csv')
train_features.drop(['Unnamed: 0'], axis = 1, inplace = True)
test_features.drop(['Unnamed: 0'], axis = 1, inplace = True)

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
    preddf = preddf.sort('img')
    imgs = []
    coverage = []
    overlap2 = []
    overlap3 = []
    imgprev = preddf.img.values[0]
    mat = np.zeros((544, 544))
    for c, row in tqdm(preddf.iterrows(), miniters=100):
        if row['img'] != imgprev:
            coverage.append(np.sum(mat>0))
            overlap2.append(np.sum(mat>1))
            overlap3.append(np.sum(mat>2))
            imgs.append(row['img'])
            mat = np.zeros((544, 544))
            imgprev = row['img']
        row_idx = np.array(range(int(row['x0']), int(row['x1'])))   
        col_idx = np.array(range(int(row['y0']), int(row['y1'])))
        mat[row_idx[:, None], col_idx] = mat[row_idx[:, None], col_idx] + 1
    sealCover = pd.DataFrame({'img': imgs, 'sealCoverage': coverage, 'sealOverlap2': overlap2, 'sealOverlap3': overlap3})
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

resn50CVOlap.head()


## Prepare data
#X = np.array(train_features.drop(['image_name', 'tags'], axis=1))
#y_train = []
#
#flatten = lambda l: [item for sublist in l for item in sublist]
#labels = np.array(list(set(flatten([l.split(' ') for l in train_features['tags'].values]))))
#
#label_map = {l: i for i, l in enumerate(labels)}
#inv_label_map = {i: l for l, i in label_map.items()}
#
#for tags in tqdm(train.tags.values, miniters=1000):
#    targets = np.zeros(17)
#    for t in tags.split(' '):
#        targets[label_map[t]] = 1 
#    y_train.append(targets)
#    
#y = np.array(y_train, np.uint8)
#
#print('X.shape = ' + str(X.shape))
#print('y.shape = ' + str(y.shape))
#
#n_classes = y.shape[1]
#
#X_test = np.array(test_features.drop(['image_name', 'tags'], axis=1))
#
## Train and predict with one-vs-all strategy
#y_pred = np.zeros((X_test.shape[0], n_classes))
#
#print('Training and making predictions')
#for class_i in tqdm(range(n_classes), miniters=1): 
#    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, \
#                              silent=True, objective='binary:logistic', nthread=-1, \
#                              gamma=0, min_child_weight=1, max_delta_step=0, \
#                              subsample=1, colsample_bytree=1, colsample_bylevel=1, \
#                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, \
#                              base_score=0.5, seed=random_seed, missing=None)
#    model.fit(X, y[:, class_i])
#    y_pred[:, class_i] = model.predict_proba(X_test)[:, 1]
#
#preds = [' '.join(labels[y_pred_row > 0.2]) for y_pred_row in y_pred]
#
#subm = pd.DataFrame()
#subm['image_name'] = test_features.image_name.values
#subm['tags'] = preds
#subm.to_csv('submission.csv', index=False)
