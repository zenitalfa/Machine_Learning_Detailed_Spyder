# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 22:38:12 2021

@author: zenit
"""

"""
IMPORTANT!!!!!!!!!!!!!

## This bunch of codes will include Linear Regression, Polynomial Regression, 

SVR(Support Vector Regression), Decision Tree Regression and also lastly

Random Forest.

## R2 Score method is also included

## Be aware of that before using


"""


# =============================================================================
# DATA AND LIBRARIES IMPORT 
# =============================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn


dataSet = pd.read_csv('maaslar_polynomial.csv') # data is imported


print(dataSet.head(10))

sbn.scatterplot(x= "Egitim Seviyesi", y= "maas", data = dataSet)
sbn.lineplot(dataSet["Egitim Seviyesi"], dataSet["maas"] )
plt.show()


# sbn.displot(dataSet)


x = dataSet.iloc[:,1:2].values # np array
print(x)

y = dataSet.iloc[:,-1:].values # np array
print(y)

print(dataSet.describe())
# =============================================================================
# Linear Regression
# =============================================================================


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x,y) # learn y from x

guess_linear = lin_reg.predict(x)

# Visualising

plt.scatter(x, guess_linear, color="black")
plt.plot(x, guess_linear, color= "green")
plt.show()

# =============================================================================
# Polynomial Regression - degree= 4
# =============================================================================

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

x_poly = poly_reg.fit_transform(x)
print(x_poly)

lin_reg.fit(x_poly,y) # learn y from x_polynomial

guess_polynomial = lin_reg.predict(x_poly) # guess_2 converted to polynomial

print(guess_polynomial)

# Visualising Polynomial

plt.scatter(x,y, color="black")
plt.plot(x, guess_polynomial, color="orange")
plt.xlabel("Eğitim Seviyesi")
plt.ylabel("Maaş")
plt.title("Polynomial Regression")
plt.legend(['Polynomial Regression','Data points'])
plt.show()

# Random guess 

lin_reg.predict(poly_reg.fit_transform([[5]]))


# =============================================================================
# SVM Regression- Support Vector Machine
# =============================================================================

"""
# We should scale inputs in order to use SVR model
# StandarScaler should be imported from sklearn prepross.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Scaling

scaler = StandardScaler()

svr_reg = SVR(kernel = 'rbf', gamma='scale')

x_scaled = scaler.fit_transform(x)
print(x_scaled)

y_scaled = scaler.fit_transform(y)
print(y_scaled)

# SVR

svr_reg.fit(x_scaled, y_scaled) #learn y_scaled from x_scaled

guess_svr = svr_reg.predict(x_scaled)
print(guess_svr)


# Visualing

plt.scatter(x_scaled, y_scaled, color= "black", marker= "*")
plt.plot(x_scaled, guess_svr, color="orange")
plt.xlabel("Eğitim Seviyesi")
plt.ylabel("Maaş")
plt.title("Support Vector Regression(SVR)")
plt.legend(["SVR","Data points"])
plt.show()

# random prediction

svr_reg.predict([[5]])


# =============================================================================
# DECISION TREE 
# =============================================================================

from sklearn.tree import DecisionTreeRegressor

decision_tree_reg = DecisionTreeRegressor(random_state=0)

decision_tree_reg.fit(x,y)

guess_decision_tree = decision_tree_reg.predict(x)
print(guess_decision_tree)


# Visualising

plt.scatter(x,y, color="black")
plt.plot(x,guess_decision_tree, color="orange")
plt.xlabel("Eğitim Seviyesi")
plt.ylabel("Maaş")
plt.title("Decision Tree Regression")
plt.legend(["Decision Tree Regression","Data Points"])
plt.show()



# =============================================================================
# RANDOM FOREST REGRESSİON
# =============================================================================


from sklearn.ensemble import RandomForestRegressor

random_forest_reg = RandomForestRegressor(n_estimators=10, random_state=0)

# n_estimators is the number of trees in forest

random_forest_reg.fit(x,y.ravel()) # learn y from x

guess_random_forest = random_forest_reg.predict(x)
print(guess_random_forest)

print(random_forest_reg.predict([[5]])) #random prediction

# Visualising
z = x + 0.5
k = x-0.4

plt.scatter(x,y ,color="red")
plt.plot(x, guess_random_forest, color="blue")
plt.plot(x, random_forest_reg.predict(z), color="orange")
plt.xlabel("Eğitim Seviyesi")
plt.ylabel("Maaş")
plt.title("Random Forest Regression")
plt.legend(["Random Forest Regression","Z points ", "Data Points"])






# =============================================================================
# R2 SCORE
# =============================================================================

"""
The best practice is to get R=1 but there is an option for Decision Tree as it sort data out
small portions before prediction and there is no average approximation as in Random Forest

Decision Tree specifies the y(independent value) precisely but it is not trustworthy as it seems

"""

from sklearn.metrics import r2_score

## Y is our independent value that we try to predict
## y_guess can be product any predcition result(polynomial,SVR, Decision Tree etc)


R_square_rf = r2_score(y, guess_random_forest) # random forest R-square
print(f"Random Forest R square: {R_square_rf}")

print("---------------------")
R_square_linear = r2_score(y, guess_linear) # linear R-square
print(f"Linear Regression R square : {R_square_linear}")
print("---------------------")

R_square_poly = r2_score(y, guess_polynomial) # polynomial R-square
print(f"Polynomial Regression R square : {R_square_poly}")
print("---------------------")

R_square_dt = r2_score(y, decision_tree_reg.predict(x)) # Decision tree R-square
print(f"Decision Tree R square : {R_square_dt}")
print("---------------------")

R_square_svr = r2_score(y_scaled, svr_reg.predict(x_scaled))
print(f"SVR R square :{R_square_svr}")
print("---------------------")













