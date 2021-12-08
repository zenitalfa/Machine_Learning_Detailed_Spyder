# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 22:40:42 2021

@author: zenit
"""

# =============================================================================
#   POLYNOMİAL REGRESSİON MODEL 
# =============================================================================


#########################
######## import libraries #####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn # seaborn is fancy visualising lib


dataset = pd.read_csv("maaslar_polynomial.csv")

print(dataset)

## Scatterplot is quite useful graph for understanding data

sbn.scatterplot(x="Egitim Seviyesi", y= "maas",data=dataset, 
                color="red")

## we can find out whether there are null input or not

print(dataset.isnull())

## head

print(dataset.head(10))


# =============================================================================
# Data is separated by X and Y
# =============================================================================

x = dataset.iloc[:,1:-1].values
print(x)


y = dataset.iloc[:, -1:].values
print(y)

type(y) ## this demonstrates that x and y are numpy array



# =============================================================================
# Linear Regression
# =============================================================================


from sklearn.linear_model import LinearRegression

regressor = LinearRegression() # object is created

regressor.fit(x, y) # learning y from x

guess = regressor.predict(x)

# =============================================================================
# Visualising- Linear Regression
# =============================================================================

plt.plot(x,guess, color="blue", linewidth =5)
plt.show()



# =============================================================================
# Polynomial Regression -4th degree
# Nonlinear Regression
# =============================================================================

from sklearn.preprocessing import PolynomialFeatures # class created

poly_reg = PolynomialFeatures(degree=4) # object created

# when degree has increased, then it predicts better
# degree is adjustable

x_poly =poly_reg.fit_transform(x) # polynomial values 

print(x_poly) # x and x^2 (degree=2) printed

regressor2 = LinearRegression()

regressor2.fit(x_poly, y)

guess_2 = regressor2.predict(x_poly) # guess_2 converted to polynomial also


# =============================================================================
# Visualising- Polynomial Regression
# =============================================================================

plt.scatter(x,y, color="red")
plt.plot(x, guess_2, color="black")

plt.show()

## Burada dikkat edilmesi gereken husus
## Sonucu çizdirmeden önce bunun PolynomialFeatures'a tabi tutulması
## Yani guess_2, x_poly kullanılarak predict edilmesi  


# =============================================================================
# Guesssing random X values
# =============================================================================

regressor.predict([[6.6]]) ## guess for linear regression

regressor2.predict(poly_reg.fit_transform([[6.6]])) ## guess with polynomial regression












