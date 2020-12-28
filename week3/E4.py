# Imports 
import time
import numpy as np
from skimage.io import imread_collection
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#%%
root = '/home/tuomas/Python/DATA.ML.200/Ex3/'

images = []
labels = []
for i in range(0,9):
    fn = '0000{}/*.jpg'.format(i)
    print(fn)
    imgs = imread_collection(root + fn)
    images.append(np.array(imgs, dtype='object'))
    labels.append( np.ones(len(imgs)) * i )
    
#%%
images = np.array(images, dtype='object')
counts = [img.shape[0] for img in images]

#%% Scale all images to the range 0...1
images = np.concatenate(images)
images /= 255.
labels = np.concatenate(labels).astype('uint8')
#%% Resize all images to 32 x 32
images_resized = []
for img in images:
    images_resized.append(resize(img, (32,32)))

images_resized = np.array(images_resized)

#%% Resize all images to 32 x 32 + flatten
images_resized = []
for img in images:
    images_resized.append(cv2.resize(img, (32,32)).flatten())

images_resized = np.array(images_resized)

#%% Concatenate train and test set from previous exercise
trainX_ = np.concatenate((trainX, trainY))



#%% Best model = Random forest classifier
model_rf = RandomForestClassifier(n_estimators=20)

#%%
cv_scores = cross_val_score(model_rf, )











