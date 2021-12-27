# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:32:08 2021

@author: Yiğit Durdu

contact: yigit.durdu@gmail.com

"""

"""
This bunch of codes also includes:
    Confusion Matrix
    K- Nearest Neighbourhood( K-NN algorithm)


"""
# =============================================================================
# DATA & LIBRARY IMPORT
# =============================================================================
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sbn
import sklearn


dataSet = pd.read_csv('veriler.csv')


x = dataSet.iloc[:,1:4].values
print(x)

y = dataSet.iloc[:,-1:].values
print(y)


# Visualising

fig= plt.figure(dpi=200)

axes1= fig.add_subplot(1,1,1, projection='3d')
sbn.set(style="darkgrid")


boy = dataSet["boy"]
kilo = dataSet["kilo"]
yas = dataSet["yas"]



axes1.scatter(boy,kilo,yas, color="red")
axes1.set_title("boy-kilo-yas scatter")
axes1.set_xlabel("Boy")
axes1.set_ylabel("Kilo")
axes1.set_zlabel("Yaş")



# =============================================================================
# DATA SPLITTING
# =============================================================================


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state= 0) 




# =============================================================================
# SCALING
# =============================================================================

from sklearn.preprocessing import StandardScaler

scaler_object = StandardScaler()

# Split data scaled

X_train = scaler_object.fit_transform(x_train)
X_test = scaler_object.transform(x_test)

# no need to learn from x_test, that is why "fit" method is not included


# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=0)

log_reg.fit(X_train, y_train) # y_train is learned from X_train(scaled)

guess_logistic = log_reg.predict(X_test)



# =============================================================================
#  CONFUSION MATRİX
# further reading
# https://machinelearningmastery.com/confusion-matrix-machine-learning/
# =============================================================================


from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(y_test, guess_logistic)

print("Logistic Regression")
print(confusion_mat)




# =============================================================================
# K-NN Algorithm
# =============================================================================

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors= 5, metric= 'minkowski')

neigh.fit(X_train, y_train) # it learns y_train from X_train(scaled)

knn_guess = neigh.predict(X_test)

confusion_knn = confusion_matrix(y_test, knn_guess)

"""
Neighbors' quantity doesn't provide the better prediction
Now we have only one correct prediction according to 5 neighbors

and let have it 1 neighbor
"""



neigh2 = KNeighborsClassifier(n_neighbors= 5, metric= 'minkowski')

neigh2.fit(X_train, y_train)

knn_guess2 = neigh2.predict(X_test)

confusion_knn_2 = confusion_matrix(y_test, knn_guess2)

print("K-Nearest Neighbor Classification")
print(confusion_knn_2)

## İn this case there is only one wrong prediction

# =============================================================================
# SVM Classification
# =============================================================================


from sklearn.svm import SVC

svm_classification = SVC(kernel='rbf')

svm_classification.fit(X_train,y_train)

svm_guess = svm_classification.predict(X_test)

conf_svm = confusion_matrix(y_test, svm_guess)

print("Supported Vector Classification")
print(conf_svm)



# =============================================================================
# NAIVE BAYES- GAUSSIAN
# =============================================================================


from sklearn.naive_bayes import GaussianNB

clf= GaussianNB()


clf.fit(X_train, y_train)

bayes_pred = clf.predict(X_test)


# =============================================================================
# cONFUSION MATRIX
# =============================================================================


from sklearn.metrics import confusion_matrix

matrix_bayes = confusion_matrix(y_test, bayes_pred)


print("Gaussion Naive Bayes")
print(matrix_bayes)


# =============================================================================
# DECISION TREE CLASSIFIER
# =============================================================================

from sklearn.tree import DecisionTreeClassifier

dtree_class = DecisionTreeClassifier(criterion = "entropy")

dtree_class.fit(X_train,y_train)

d_tree_guess = dtree_class.predict(X_test)


matrix_dtree = confusion_matrix(y_test, d_tree_guess)

print("Decision Tree Classifier")
print(matrix_dtree)


# =============================================================================
# Random Forest Classification
# =============================================================================


from sklearn.ensemble import RandomForestClassifier


rForest_classifier = RandomForestClassifier(n_estimators=10, criterion = "entropy")

rForest_classifier.fit(X_train, y_train)

RF_guess = rForest_classifier.predict(X_test)

matrix_RF_clf = confusion_matrix(y_test, RF_guess)

print("Random Forest Classifier Confusion Matrix")
print(matrix_RF_clf)


# =============================================================================
#   Probabilities of Prediction for any method
# =============================================================================

y_proba = rForest_classifier.predict_proba(X_test)

print(y_proba)

# =============================================================================
# ROC(Receiver Operating Characteristics)
# =============================================================================


from sklearn import metrics

fpr, tpr, threshold = metrics.roc_curve(y_test, y_proba[:,0], pos_label='e')

print(fpr)

print(tpr)


metrics.plot_roc_curve(rForest_classifier, X_test, y_test) 










