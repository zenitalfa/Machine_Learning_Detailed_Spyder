# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
This bunch of codes covers the Multiple Linear Regression, Decision Tree Reg
Random Forest Reg and Polynomial Regression with R-square values

This comparison will result in understanding of these prediction methods

Yiğit Durdu

Contact:
    
yigit.durdu1@gmail.com

"""

# =============================================================================
#   DATA AND LIBRARIES IMPORT 
# =============================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import r2_score


dataSet = pd.read_csv('maaslar_yeni.csv') 

unvan = dataSet.iloc[:,2:3].values
kidem = dataSet.iloc[:,3:4].values
puan = dataSet.iloc[:,4:5].values
maas = dataSet.iloc[:,-1:].values

dataSet.corr() # prints correlation between parameters

# =============================================================================
# Before preprocessing data, better to review data from many aspects
# İt may help us to code efficiently
# =============================================================================
fig, axes = plt.subplots(2,2)

axes[0,0].scatter(kidem,puan)
axes[0,0].set_title("Kıdem-Puan Scatter")


axes[0,1].scatter(kidem,maas)
axes[0,1].set_title("Kidem-Maaş Scatter")

axes[1,0].scatter(puan,maas)
axes[1,0].set_title("Puan-Maaş Scatter")



fig = plt.figure(dpi=200)

ax1 = fig.add_subplot(1,2,1 ,projection='3d')
ax2 = fig.add_subplot(1,2,2, projection='3d')

ax1.scatter(kidem, puan, maas)
ax1.set_xlabel("Kıdem Durumu")
ax1.set_ylabel("Puan Durumu")
ax1.set_zlabel("Maaş Durumu")


ax2.scatter(maas,puan,kidem)
ax2.set_xlabel("Maaş Durumu")
ax2.set_ylabel("Puan Durumu")
ax2.set_zlabel("Kıdem Durumu")



# =============================================================================
# fig, ax = plt.subplots(2)
# 
# ax[0].plot(kidem,puan)
# ax[1].scatter(puan,maas)
# ax[1].plot(puan,maas)
# =============================================================================

# =============================================================================
# Data Splitting- x_train,x_test, y_train, y_test
# =============================================================================

""" 

y = a.x1 + b.x2+ c.x3+ error can be defined as Multiple Regression model

In our case, "maas" column has been selected as y(independent value) and other columns are x1, x2...

However, we need to split the date as train and test data.

İf there was any requirements about encoding( Label or One Hot Encoding), it was going to be done

But we have numerical inputs or categorical inputs are just like labels or index


"""


pre_dataSet = dataSet.iloc[:,2:-1]

maas_frame= dataSet.iloc[:,-1:]


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(pre_dataSet, maas_frame, 
                                                    test_size=0.33, random_state = 0)



# =============================================================================
# ----------------------------Model Construction------------------------
# 
#
#
# from sklearn.linear_model import LinearRegression
#
# =============================================================================

from sklearn.linear_model import LinearRegression # class imported

lin_reg= LinearRegression() # object has been called

lin_reg.fit(x_train, y_train)

## Model learn y_train from x_train

lin_reg.predict(x_test)




# =============================================================================
# -------------------------BACKWARD ELIMINATION-----------------
# 
# We will use backward elimination model in order to realize which inputs/coloumns
# do affect model relatively in a way negative
# =============================================================================

import statsmodels.api as sm


X= np.append(arr= np.ones((30,3)).astype(int), values= pre_dataSet, axis = 1) 

## We convert our split dataset to numpy array in order to get P value

X_list = pre_dataSet.iloc[:, [0,1,2]].values
X_list = np.array(X_list, dtype=float)

## We will eliminate irrelated columns by using this X_list


model = sm.OLS(maas,X_list).fit()
print(model.summary())

# =============================================================================
#  According to P values, can be seen that x2 and x3 values
# have rather high P values (0.998 and 0.710)
# =============================================================================

X_list = pre_dataSet.iloc[:, [0,2]].values
X_list = np.array(X_list, dtype=float)
model = sm.OLS(maas,X_list).fit()
print(model.summary())


X_list = pre_dataSet.iloc[:, [0]].values
X_list = np.array(X_list, dtype=float)
model = sm.OLS(maas,X_list).fit()
print(model.summary())


x_train = x_train.iloc[:,0:1]
x_test = x_test.iloc[:,0:1]

lin_reg.fit(x_train, y_train)

guess = lin_reg.predict([[]]) 

























