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
from sklearn.preprocessing import scale, StandardScaler

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

#%% Preprocess the images
imgs_processed = []
scaler = MinMaxScaler()
for img in images:
    # Resize & vectorize the image
    img = cv2.resize(img, (32,32)).ravel()
    # Scale sample to (0,1)
    img_T = img[...,None]
    scaler.fit(img_T)
    img_T = scaler.transform(img_T)
    
    imgs_processed.append(img)
    
imgs_processed = np.array(imgs_processed).astype('float32')
# Normalize the data
# Scale sample to (0,1)
    # Since MinMaxScaler scales data featurewise, I transposed the row vector into column vector
    # so the samplewise scaling is performed

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

#%% Create training and testing sets
trainX, testX, trainY, testY = train_test_split(imgs_processed, 
                                                labels, 
                                                test_size=0.3)
                                                #random_state=0)
                                                
#%%
scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)     
#scaler.fit(testX)
#testX = scaler.transform(testX)                                     
#%% Train, test & evaluate given models
models = [KNeighborsClassifier(n_neighbors=3),
          LinearDiscriminantAnalysis(solver='svd'),
          LogisticRegression(max_iter=10000),
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
    
#%% Print the statistics
model_names = ['3-NN','LDA','LogReg','SVM linear','SVM rbf','Random forest']
print('Training set size (batch) = {}'.format(trainX.shape[0]))
print('Test set size = {}\n'.format(testX.shape[0]))
for i in range(6):
    print('Results for {}'.format(model_names[i]))
    print('----------------------')
    print('Accuracy is {} %'.format(round(accuracy[i]*100, 2)))
    print('Training time / batch {} s'.format(round(tr_time[i], 2)))
    print('Test time / sample {} ms\n'.format(round(tst_time[i]*1000/testX.shape[0], 3)))
    
#%%
''' Task 4 starts '''
#%% Concatenate train and test set from previous exercise
from sklearn.model_selection import cross_val_score
trainX2 = np.concatenate((trainX, testX))
trainY2 = np.concatenate((trainY, testY))

model_best = RandomForestClassifier(n_estimators=20)
cv_scores = cross_val_score(models[3], trainX2, trainY2, cv=5, verbose=2)
#%%
print('Mean of accuracies = {}'.format(np.mean(cv_scores)))
print('Variance of accuracies = {}'.format(np.var(cv_scores)))

#%%
''' Task 4 starts '''
#%% Resize all images to 32 x 32
images_resized = []
for img in images:
    images_resized.append(resize(img, (32,32)))

images_resized = np.array(images_resized)

#%% Create training and testing sets.
from tensorflow.keras.utils import to_categorical
trainX, testX, trainY, testY = train_test_split(images_resized, 
                                                labels, 
                                                test_size=0.15)

# One-hot encoding
trainY = to_categorical(trainY, num_classes=9)
testY_cat = to_categorical(testY, num_classes=9)
#%% Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,BatchNormalization,Dropout,Flatten,Dense

layers=[Conv2D(filters=32, kernel_size = 3, activation='relu', input_shape = (32,32,3)),
        BatchNormalization(),
        Conv2D(filters=32, kernel_size = 3, activation='relu'),
        BatchNormalization(),
        Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Conv2D(64, kernel_size = 3, activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size = 3, activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Conv2D(128, kernel_size = 4, activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dropout(0.4),
        Dense(9, activation='softmax')]

cnn_model = Sequential(layers)
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()

#%% Train the model
cnn_model.fit(trainX, trainY, epochs=10, batch_size=64, verbose=1)

#%% Evaluate the model
test_loss, test_acc = cnn_model.evaluate(testX, testY_cat, verbose=2)
print("Accuracy of CNN = {}".format(test_acc))



























