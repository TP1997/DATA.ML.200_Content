import numpy as np
import scipy.io

# a) Load data & split into training & testing samples
root = '/home/tuomas/Python/DATA.ML.200/Ex2/'
data = scipy.io.loadmat(root + 'twoClassData.mat')
dataX = data['X']
dataY = data['y'][0]

trainX = dataX[:200,]
trainY = dataY[:200]

testX = dataX[200:,]
testY = dataY[200:]

#%%
# 3NN-classifier
from sklearn.neighbors import KNeighborsClassifier

K = 3
KNN = KNeighborsClassifier(n_neighbors=K)
# Train the model using the training sets
KNN.fit(trainX, trainY)
KNN_predY = KNN.predict(testX)
KNN_probs = KNN.predict_proba(testX)

KNN_acc = np.sum(testY==KNN_predY) / testY.size
print("{}NN-accuracy: {}".format(K, KNN_acc))

#%%
# Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

LDA = LinearDiscriminantAnalysis(solver='svd')
# Train the model using the training sets
LDA.fit(trainX, trainY)
LDA_predY = LDA.predict(testX)
LDA_probs = LDA.predict_proba(testX)

LDA_acc = np.sum(testY==LDA_predY) / testY.size
print("LDA-accuracy: {}".format(LDA_acc))

#%%
# Logistic Regression
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
# Train the model using the training sets
LR.fit(trainX, trainY)
LR_predY = LR.predict(testX)
LR_probs = LR.predict_proba(testX)

LR_acc = np.sum(testY==LR_predY) / testY.size
print("LR-accuracy: {}".format(LR_acc))

#%%
# Random Forest
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20)
# Train the model using the training sets
RF.fit(trainX, trainY)
RF_predY = RF.predict(testX)
RF_probs = RF.predict_proba(testX)

RF_acc = np.sum(testY==RF_predY) / testY.size
print("RF-accuracy: {}".format(RF_acc))

#%%
# ROC & AUC for each classifier
from sklearn import metrics

KNN_fpr, KNN_tpr, th = metrics.roc_curve(testY, KNN_probs[:,1])
KNN_auc = metrics.auc(KNN_fpr, KNN_tpr)

LDA_fpr, LDA_tpr, th = metrics.roc_curve(testY, LDA_probs[:,1], pos_label=1)
LDA_auc = metrics.auc(LDA_fpr, LDA_tpr)

LR_fpr, LR_tpr, th = metrics.roc_curve(testY, LR_probs[:,1], pos_label=1)
LR_auc = metrics.auc(LR_fpr, LR_tpr)

RF_fpr, RF_tpr, th = metrics.roc_curve(testY, RF_probs[:,1], pos_label=1)
RF_auc = metrics.auc(RF_fpr, RF_tpr)

#%%
import matplotlib.pyplot as plt
#from matplotlib.pyplot.figure as figure

plt.figure(figsize=(10, 10))
plt.title('Receiver Operating Characteristic')
plt.plot(KNN_fpr, KNN_tpr, '-b', label = '3NN AUC = {}'.format(np.round(KNN_auc,2)))
plt.plot(LDA_fpr, LDA_tpr, '-g', label = 'LDA AUC = {}'.format(np.round(LDA_auc,2)))
plt.plot(LR_fpr, LR_tpr, '-r', label = 'LR AUC = {}'.format(np.round(LR_auc,2)))
plt.plot(RF_fpr, RF_tpr, '-c', label = 'RF AUC = {}'.format(np.round(RF_auc,2)))

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'--k')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#%%
import matplotlib.pyplot as plt
import pandas as pd

col_names = ['Model', 'Accuracy Score', 'ROC-AUC Score']
names = ['3-Nearest Neighbor','Linear Discriminant Analysis','Logistic Regression','Random Forest']
accs = [KNN_acc, LDA_acc, LR_acc, RF_acc]
roc_aucs = np.round([KNN_auc, LDA_auc, LR_auc, RF_auc], 2)

df = pd.DataFrame({'Model':names, 'Accuracy Score':accs, 'ROC-AUC Score':roc_aucs})
plt.table(df)
#%%
# b) Fill in the attached table
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

col_names = ['Model', 'Accuracy Score', 'ROC-AUC Score']
names = ['3-Nearest Neighbor','Linear Discriminant Analysis','Logistic Regression','Random Forest']
accs = [KNN_acc, LDA_acc, LR_acc, RF_acc]
roc_aucs = np.round([KNN_auc, LDA_auc, LR_auc, RF_auc], 2)

df = pd.DataFrame({'Model':names, 'Accuracy Score':accs, 'ROC-AUC Score':roc_aucs})

ax.table(cellText=df.values, colLabels=df.columns, loc='center')

fig.tight_layout()
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig(root+'table.png', dpi=300)

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))

ax.table(cellText=df.values, colLabels=df.columns, loc='center')

fig.tight_layout()

fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig(root+'table.png', dpi=300)

















 
