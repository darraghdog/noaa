
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from PIL import Image
import gc, math
import pickle

from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix


from cnnmodels import vgg_std16_model, preprocess_input, create_rect5, load_img, train_generator, test_generator
from cnnmodels import identity_block, testcv_generator, conv_block, resnet50_model


# In[2]:

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


# In[3]:

# Params
img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_class = 2
ROWS, COLS = 224, 224
BATCHSIZE = 64 # 128
SEAL_CLASSES = ['seals_0', 'seals_1', 'seals_2_4', 'seals_5_10', 'seals_11+']
nb_perClass = int(BATCHSIZE / len(SEAL_CLASSES))
TRAIN_DIR = '../darknet/seals/JPEGImagesBlk'
TEST_DIR = '../darknet/seals/JPEGImagesTest'
num_class = len(SEAL_CLASSES)


# In[4]:

# CV object detections
rfcnCVtmp = pd.read_pickle('../coords/rfcnmultiCV.pkl')
# Read in the previous preds 
dftmp = pd.concat([pd.read_csv('../coords/vggCVPreds2604_fold2.csv'),
                pd.read_csv('../coords/vggCVPreds2604_fold1.csv')])
dftmp.columns = ['img1', 'predSeal']
rfcnCV = pd.concat([rfcnCVtmp.reset_index(drop=True), dftmp.reset_index(drop=True)], axis=1)
rfcnCV = rfcnCV[rfcnCV['predSeal']>0.2].reset_index(drop=True)
rfcnCV.head()


# In[5]:

# Add on the classes for bins
rfcnCV['seal_cut'] = pd.cut(rfcnCV['seal'], bins = [-1,0,1,4,10,30])
rfcnCV['seals_0'] = np.where(rfcnCV['seal']==0, 1, 0)
rfcnCV['seals_1'] = np.where(rfcnCV['seal'] == 1, 1, 0)
rfcnCV['seals_2_4'] = np.where(rfcnCV['seal'].between(2,4), 1, 0)
rfcnCV['seals_5_10'] = np.where(rfcnCV['seal'].between(5,10), 1, 0)
rfcnCV['seals_11+'] = np.where(rfcnCV['seal']>10, 1, 0)
rfcnCV.head(5)


# In[6]:

# Test object detections
rfcnTsttmp = pd.read_pickle('../coords/rfcnmultiTst.pkl')
# Read in the previous preds 
dftmp = pd.read_csv('../coords/vggTestPreds2604.csv')
dftmp.columns = ['img1', 'predSeal']
rfcnTst = pd.concat([rfcnTsttmp.reset_index(drop=True), dftmp.reset_index(drop=True)], axis=1)
rfcnTst = rfcnTst[rfcnTst['predSeal']>0.2].reset_index(drop=True)
print(rfcnTst.shape)
rfcnTst.head()


# In[7]:

for c, row in rfcnCV.iterrows():
    if c==5:break
    print row.tolist()[10:]


# In[8]:

def train_generator(datagen, df):
    while 1:
        batch_x = np.zeros((BATCHSIZE, ROWS, COLS, 3), dtype=K.floatx())
        batch_y = np.zeros((BATCHSIZE, len(SEAL_CLASSES)), dtype=K.floatx())
        fn = lambda obj: obj.loc[np.random.choice(obj.index, size=nb_perClass, replace=False),:]
        batch_df = df.groupby(['seal_cut'], as_index=True).apply(fn)
        i = 0
        for index,row in batch_df.iterrows():
            row = row.tolist()
            image_file = os.path.join(TRAIN_DIR, row[0])
            seal = row[6]
            bbox = row[2:6]
            cropped = load_img(image_file+'.jpg',bbox,target_size=(ROWS,COLS))
            x = np.asarray(cropped, dtype=K.floatx())
            x = datagen.random_transform(x)
            x = preprocess_input(x)
            batch_x[i] = x
            batch_y[i] = row[10:] # Add in all classes
            i += 1
        yield (batch_x.transpose(0, 3, 1, 2), batch_y)
        #return (batch_x.transpose(0, 3, 1, 2), batch_y)

def test_generator(df, datagen = None, batch_size = BATCHSIZE):
    n = df.shape[0]
    batch_index = 0
    while 1:
        current_index = batch_index * batch_size
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1    
        else:
            current_batch_size = n - current_index
            batch_index = 0        
        batch_df = df[current_index:current_index+current_batch_size]
        batch_x = np.zeros((batch_df.shape[0], ROWS, COLS, 3), dtype=K.floatx())
        i = 0
        for index,row in batch_df.iterrows():
            row = row.tolist()
            image_file = os.path.join(TEST_DIR, row[0]+'.jpg')
            bbox = row[2:6]
            cropped = load_img(image_file,bbox,target_size=(ROWS,COLS))
            x = np.asarray(cropped, dtype=K.floatx())
            if datagen is not None: x = datagen.random_transform(x)            
            x = preprocess_input(x)
            batch_x[i] = x
            i += 1
        if batch_index%16000 == 0: print(batch_index)
        #return(batch_x.transpose(0, 3, 1, 2))
        yield(batch_x.transpose(0, 3, 1, 2))
        
# Data generator
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True)


# In[9]:

# Lets make our validation set
CVsplit = rfcnCV.img.str.split('_').apply(lambda x: x[0]).astype(int) % 40 == 0
train_df = rfcnCV[~CVsplit]
valid_df = rfcnCV[CVsplit]
test_df = rfcnTst


# In[10]:

test_df.head()


# In[11]:

# validation_data (valid_x,valid_y)
df_1 = valid_df
l = valid_df.groupby('seal_cut').size() 
nb_NoF_valid = math.ceil(l.sum()/10)
valid_x = np.zeros((valid_df.shape[0], ROWS, COLS, 3), dtype=K.floatx())
valid_y = np.zeros((valid_df.shape[0], len(SEAL_CLASSES)), dtype=K.floatx())
i = 0
for index,row in valid_df.iterrows():
    row = row.tolist()
    image_file = os.path.join(TRAIN_DIR, row[0])
    seal = row[6]
    bbox = row[2:6]
    cropped = load_img(image_file+'.jpg',bbox,target_size=(ROWS,COLS))
    x = np.asarray(cropped, dtype=K.floatx())
    x = preprocess_input(x)
    valid_x[i] = x
    valid_y[i] = row[10:]
    i += 1
valid_x = valid_x.transpose(0, 3, 1, 2)
valid_x.shape


# In[12]:

# Load our model
nb_epoch = 2
samples_per_epoch = 70400
model = vgg_std16_model(ROWS, COLS, channel, num_class)

# Start Fine-tuning
model.fit_generator(train_generator(train_datagen, train_df),
          nb_epoch=nb_epoch,
          samples_per_epoch=samples_per_epoch, #50000,
          verbose=1,
          validation_data=(valid_x, valid_y),
          )


# In[13]:

for layer in model.layers[10:]:
    layer.trainable = True
model.optimizer.lr = 1e-4
nb_epoch = 3
model.fit_generator(train_generator(train_datagen, df=train_df),
          nb_epoch=nb_epoch,
          samples_per_epoch=samples_per_epoch,
          verbose=1,
          validation_data=(valid_x, valid_y),
          )


# In[16]:

# test_preds = test_model.predict_generator(test_generator(test_df), val_samples=test_df.shape[0]))
test_preds = model.predict_generator(test_generator(test_df), val_samples=test_df.shape[0])


# In[17]:

df = pd.concat([test_df, pd.DataFrame(test_preds,  columns=SEAL_CLASSES)], axis=1)
df.to_pickle('../coords/vggTestMultiPreds1605.pkl')


# In[27]:

df[[0]+range(8, 13)].to_csv('../coords/vggTestMultiPreds1605.csv', index=False)
df.head(5)

