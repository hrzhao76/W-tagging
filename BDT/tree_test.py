from array import array

import h5py
import math

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc


train_file = h5py.File("/Volumes/MacOS/Research/Data/high-level/train_no_pile_10000000.h5", "r")
test_file = h5py.File("/Volumes/MacOS/Research/Data/high-level/test_no_pile_5000000.h5", "r")

X_train = train_file['features'][()]
y_train = train_file['targets'][()]
X_test = test_file['features'][()]
y_test = test_file['targets'][()]

N = 100000
X_train = X_train[0:N,]
y_train = y_train[0:N]
X_test = X_test[0:N,]
y_test = y_test[0:N]




dt = DecisionTreeClassifier(max_depth=49,
                            min_samples_leaf = math.ceil(0.25*len(X_train)),
                            min_samples_split = math.ceil(0.0021*len(X_train)))
bdt = AdaBoostClassifier(dt,
                         algorithm='SAMME',
                         n_estimators=500,
                         learning_rate=0.07)

bdt.fit(X_train, y_train.ravel())
sk_y_predicted = bdt.predict(X_test)

print(classification_report(y_test, sk_y_predicted, target_names=["background", "signal"]))
print("Area under ROC curve: %.4f"%(roc_auc_score(y_test, sk_y_predicted)))
#print("Area under ROC curve: %.4f"%(auc(y_test, sk_y_predicted)))


X_test_new_1 = []
X_test_new_2 = []

for i in range(0,N):
	if y_test[i] == 1. :
		X_test_new_1.append(X_test[i,])
		pass
	else : 
		X_test_new_2.append(X_test[i,])
	pass

plt.hist(bdt.decision_function(X_test_new_1).ravel(),
         color='r', alpha=0.5, range=(-0.9,0.9), bins=100)
plt.hist(bdt.decision_function(X_test_new_2).ravel(),
         color='b', alpha=0.5, range=(-0.9,0.9), bins=100)
plt.xlabel("scikit-learn BDT output")
plt.show()


decisions = bdt.decision_function(X_test)
# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(y_test, decisions)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))

#plt.plot(tpr, 1./ fpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
# plt.yscale("log")
# plt.show()

#plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()
plt.show()


