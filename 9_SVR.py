# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:17:52 2021

@author: zenit
"""

# =============================================================================
#   SVR(SUPPORT VECTOR REGRESSİON) 
# =============================================================================


######## DATA & LIBRARIES IMPORT ###########


"""

IMPORTANT!!!!!!!!!!!!

In order to use SVR, there is kinda obligatory to use SCALER, this SVR method is vulnerable against

unrelated values(data points), so values should be scaled in case we have SVR

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

# data imported

dataset = pd.read_csv("maaslar_polynomial.csv")

print(dataset)

sbn.scatterplot(x="Egitim Seviyesi", y="maas", data=dataset)

print(dataset.head())


x = dataset.iloc[:, 1:2].values
print(x)

y = dataset.iloc[:,-1:].values
print(y)


# =============================================================================
# Linear Regression
# =============================================================================

from sklearn.linear_model import LinearRegression

regressor1 = LinearRegression()

regressor1.fit(x,y) # learning y from x, training

guess_1 = regressor1.predict(x) # prediction
print(guess_1)


# =============================================================================
# Polynomial Regression - degree=2 ( x^2)
# =============================================================================

from sklearn.preprocessing import PolynomialFeatures

poly_reg1 = PolynomialFeatures(degree=2)

x_poly1 = poly_reg1.fit_transform(x)

print(x_poly1)

regressor2 = LinearRegression()

regressor2.fit(x_poly1, y) # learn y from x_poly1

guess_2 = regressor2.predict(x_poly1) # prediction

print(guess_2)


plt.plot(x,y, color= "red")
plt.scatter(x, guess_2, color="black")


# =============================================================================
# Polynomial Regression- Degree = 4 (x^4)
# 
# =============================================================================


from sklearn.preprocessing import PolynomialFeatures

poly_reg2 = PolynomialFeatures(degree=6)

x_poly2 = poly_reg2.fit_transform(x) 

print(x_poly2)

regressor3 = LinearRegression()

regressor3.fit(x_poly2,y) # learn y from x_poly2

guess_3 = regressor3.predict(x_poly2) # prediction y from x_poly2

print(guess_3)

regressor3.predict(poly_reg2.fit_transform([[6.6]]))


## When degree is 6, prediction is quite better

plt.scatter(x,y, color="blue")
plt.plot(x, guess_3, color= "black")
plt.show()

# =============================================================================
#  Scaler
# =============================================================================

from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()

x_scaled = scaler1.fit_transform(x)

print(x_scaled)

scaler2 = StandardScaler()

y_scaled = scaler2.fit_transform(y)
print(y_scaled)

# =============================================================================
# SVR ( Support Vector Regression) - kernel function = rbf (radial basis function)
# =============================================================================



from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf', degree= 4) # SVR object described

svr_reg.fit(x_scaled,y_scaled)

guess_svr= svr_reg.predict(x_scaled)

## Visualizing

plt.scatter(x_scaled,y_scaled, color="#8F0625")
plt.plot(x_scaled , guess_svr, color="#25928A")



## Random predictions

svr_reg.predict([[6.6]])


# =============================================================================
# SVR - kernel function = precomputed
# input should be square matrix for precomputed
# =============================================================================


from sklearn.svm import SVR

svr_reg = SVR(kernel = 'precomputed', degree= 4) # SVR object described

svr_reg.fit(x_scaled,y_scaled)

guess_svr= svr_reg.predict(x_scaled)

## Visualizing

plt.scatter(x_scaled,y_scaled, color="#8F0625")
plt.plot(x_scaled , guess_svr, color="#25928A")



## Random predictions

svr_reg.predict([[6.6]])











