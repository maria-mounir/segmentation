#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install fastapi nest-asyncio pyngrok uvicorn')


# In[2]:


#get_ipython().system('pip install python-multipart')


# In[3]:


#get_ipython().system('pip install Pillow')


# In[4]:


#get_ipython().system('pip install tensorflow')


# In[5]:


#get_ipython().system('pip install tifffile ')
#get_ipython().system('pip install imagecodecs')
#get_ipython().system('pip install imagecodecs-lite')
import numpy as np
from skimage.io import imread
import tifffile as tiff 
import imagecodecs


# In[6]:


import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow import keras
from keras.models import Model
from keras import backend as K
import math
import tempfile


# In[7]:


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis = -1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


# In[8]:


def normalizeImage(img):
    img = np.array([img])
    img = img_as_float(img)
    img = img.astype('float32')
    mean = np.mean(img)  # mean for data centering
    std = np.std(img)  # std for data normalization
    img -= mean
    img /= std
    return img

# -----------------------------------------------------------------------------
def normalizeMask(mask):
    mask = np.array([mask])
    mask = img_as_float(mask)
    mask = mask.astype('float32')
    return mask

# -----------------------------------------------------------------------------
def calculateWeights(obj_mask, bckgnd_msk):
    sum_all = np.sum(obj_mask + bckgnd_msk, dtype=np.float32) + 1  
    sum_obj = np.sum(obj_mask, dtype=np.float32)
    sum_bck = np.sum(bckgnd_msk, dtype=np.float32)
    # make sure todre is at least some contribution and not 0s
    if sum_obj < 100:   
        sum_obj = 100
    if sum_bck < 100:
        sum_bck = 100
    return np.float32(obj_mask)*np.float32(sum_bck)/np.float32(sum_all) + np.float32(bckgnd_msk)*np.float32(sum_obj)/np.float32(sum_all)


# In[9]:


from keras.models import load_model
Model = tf.keras.models.load_model('odunet5.h5',custom_objects={'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss,'jaccard_distance':jaccard_distance,'iou':iou})


# In[10]:



from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Response
import json
from skimage.io import imread
import pandas as pd
import os
from skimage.util import invert, img_as_float
from sklearn.utils import shuffle
from skimage.io import imsave
from skimage.transform import resize
import io
from glob import glob
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
async def root():
    return {'hello': 'world'}

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    fille = file.filename
    path = os.path.join(os.getcwd(), fille)
    with open(path, "wb") as f:
        f.write(await file.read())
    dir_path = os.path.dirname(path)
    file_paths = glob(os.path.join(dir_path, f'{fille}*'))
   
    
    #normalization
    fx = lambda x: imread(x, as_gray=True)
    img = map(fx,file_paths)
    fx = lambda x: np.array(x[:, 250:3750], dtype=np.float32)
    img = map(fx, img)
    fx = lambda x: resize(x, (512, 512))
    img = map(fx, img)
    fx = lambda x: normalizeImage(x)
    img = map(fx, img)
    
    prediction= Model.predict(img)
 
 # Convert the prediction to an image and encode it as bytes
    prediction_image = prediction[0]
    modelpath='/Internal storage/Download/'
    for i, k in enumerate(prediction):
        imsave(f'{modelpath}{0+i}.png', k)
   
    # Read the first prediction image as bytes
    prediction_bytes = open(f'{modelpath}0.png', 'rb').read()
    
    # Return the prediction image as a response
    return Response(content=prediction_bytes, media_type="image/png")
    #return FileResponse('/Internal storage/Download/',media_type="image/png",filename=os.path.basename('/Internal storage/Download/'))


# In[11]:


import nest_asyncio
from pyngrok import ngrok
import uvicorn

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)


# In[ ]:




