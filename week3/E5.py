# Importit
import time
import numpy as np
from skimage.io import imread_collection
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

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

#images = np.array(images, dtype='object')
images = np.concatenate(images)
labels = np.concatenate(labels).astype('uint8')

#%% Create training and testing sets
trainX, testX, trainY, testY = train_test_split(images, 
                                                labels, 
                                                test_size=0.3)

#%% Preprocess the images
imgs_processed = []
for img in images[0:10]:
    # Resize & vectorize the image
    img = cv2.resize(img, (32,32)).ravel()
    # Scale to 0 ... 1
    img /= 255.
    #transformer = Normalizer().fit([img])
    #img = transformer.transform([img])
    #imgs_processed.append(img)

imgs_processed = np.array(imgs_processed)

#%%
X = [np.array([4, 1, 2, 2]),
     np.array([1, 3, 9, 3]),
     np.array([5, 7, 5, 1])]

transformer = Normalizer().fit([X[1]]) 
xx=transformer.transform([X[1]])


#%% Scale
scaler = MinMaxScaler()
scaler.fit(imgs_processed)
imgs_processed =  scaler.transform(imgs_processed)

#%% Create training and testing sets
trainX, testX, trainY, testY = train_test_split(imgs_processed, 
                                                labels, 
                                                test_size=0.3)

#%% Train, test & evaluate given models
models = [KNeighborsClassifier(n_neighbors=3),
          LinearDiscriminantAnalysis(solver='svd'),
          LogisticRegression(max_iter=1000)
          SVC(kernel='linear'),
          SVC(kernel='rbf'),
          RandomForestClassifier(n_estimators=20)
          ]

accuracy = []
tr_time = []
tst_time = []
for model in models:
    # Training
    print('Training {} ...'.format(model.__class__.__name__))
    start = time.time()
    model.fit(trainX, trainY)
    tr_time.append( time.time() - start )
    # Testing
    print('Testing {} ...'.format(model.__class__.__name__))
    start = time.time()
    predY = model.predict(testX)
    tst_time.append( time.time() - start )
    # Evaluating
    print('Evaluating {} ...'.format(model.__class__.__name__))
    accuracy.append( accuracy_score(testY, predY) )