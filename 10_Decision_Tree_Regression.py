# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 22:41:14 2021

@author: zenit
"""

""" Decision Tree algorithm is used for classification problem mostly but

it also can be used for regression problem either. This is an implimentation of that

"""

# =============================================================================
# IMPORT DATA AND LIBRARIES 
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn


dataset = pd.read_csv('maaslar_polynomial.csv')

sbn.scatterplot(x="Egitim Seviyesi", y= "maas", data= dataset)

print(dataset.head())

x = dataset.iloc[:,1:2].values
print(x)

y = dataset.iloc[:,-1:]
print(y)

# =============================================================================
#  POLYNOMİAL REGRESSİON - just repeat
# =============================================================================


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


regressor = LinearRegression()

poly_regressor = PolynomialFeatures(degree=8)


x_poly = poly_regressor.fit_transform(x)

print(x_poly)

regressor.fit(x_poly, y)

guess_poly = regressor.predict(x_poly)

plt.scatter(x, y , color="#8F0625")
plt.plot(x, guess_poly, color= "#25928A")


print(guess_poly)

regressor.predict(poly_regressor.fit_transform([[6.6]])) 

"""

# =============================================================================
# IMPORTANT!!!!!!!!!!!
#
# ## here, in order to predict random values for verifying performance, we need to
# ## use poly_regressor.fit_transform([[value]]) 
# ## it should be converted to polynomial form
# 
# =============================================================================

"""


# =============================================================================
# DECISION TREE
# =============================================================================


from sklearn.tree import DecisionTreeRegressor

regressor_decision_tree = DecisionTreeRegressor(random_state=(0)) # onject is created

regressor_decision_tree.fit(x,y) # train phase, learn y from x

guess_decision_tree = regressor_decision_tree.predict(x)


plt.scatter(x,y, color="m")
plt.plot(x, guess_decision_tree, color="blue")


regressor_decision_tree.predict([[7]])

## Decision tree algorihm just separates values and group them and find values according to that
## that is why it demonstrates result precisely 




















