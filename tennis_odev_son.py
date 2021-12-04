# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:28:05 2021

@author: zenit
"""

# =============================================================================
#  Ödev- Play Tennis 
# =============================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## Data import

veri = pd.read_csv("odev_tenis.csv")





# =============================================================================
#  Verilerin Hazırlanması
# =============================================================================


## Play ve Windy kolonları için LABEL ENCODİNG
## Outlook kolonu için ise ONEHOTENCODİNG yapmamız gerekir


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ohe = preprocessing.OneHotEncoder()



windy =  veri.iloc[:, -2:-1].values
print(windy)

play = veri.iloc[:, -1:].values
print(play)


windy = le.fit_transform(veri.iloc[:, 3:4])

print(windy)

play = le.fit_transform(veri.iloc[:,-1])


## One Hot Encoding

outlook = veri.iloc[:, :1].values
print(outlook)


outlook = ohe.fit_transform(outlook).toarray() # one hot encoding


# =============================================================================
#  Data Framelerin birleştirilmesi
# =============================================================================



play_frame = pd.DataFrame(data = play, index = range(14), columns= ["play"])

windy_frame = pd.DataFrame(data = windy, index = range(14), columns= ["windy"])

type(windy_frame)


veriler = veri.apply(preprocessing.LabelEncoder().fit_transform)


digerVeri = pd.DataFrame(data= veri.iloc[:,1:3], index = range(14), 
                         columns= ["temperature", "humidity"])


outlook_frame = pd.DataFrame(data = outlook, index = range(14), 
                             columns= ["overcast", "rainy", "sunny"])

pw_frame = pd.concat([windy_frame, play_frame],axis = 1)

sonuc1 = pd.concat([outlook_frame, digerVeri],axis = 1)

sonuc2 = pd.concat([pw_frame,sonuc1],axis = 1)


# =============================================================================
# VERİNİN EĞİTİM VE TEST OLARAK AYRILMASI
# =============================================================================




from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(sonuc2.iloc[:,:-1], 
                                                    sonuc2.iloc[:, -1:],
                                                    test_size=0.33,
                                                    random_state=0)

### Bu noktada Humidity bağımlı değişken = y
## Humidity dışındakiler bağımsız değişken = x (Multiple Linear regression için)

# sonuc2.iloc[:,-1:] >>>> Outlook, temperature, windy, play (Bunlarla öğrenecek)
# sonuc2.iloc[:, -1:] >>>>>> Humidity (Bunu tahmin edecek)




##############################################################################
############################ Multiple Linear Regression ######################
######################################################################

################### Model İnşası ##################################
#################################################


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()


regressor.fit(x_train, y_train)

""" ÖNEMLİİİİİİİİİ
BURADA "regressor" objesinin yaptığı iş : 
    
    x_train'den (bağımsız değişkenlerinden) y_train(bağımlı değişkenini) öğrenmektir
    
    yani aralarında LİNEER BİR MODEL KURMAK anlamına gelir
    
    """
    
    ## x_train ile makine eğitildi
    ## x_test ile bir tahmin algoritması çıkarılacaktır
    

y_pred = regressor.predict(x_test)

print(y_pred)


print(y_pred)

# =============================================================================
#  BACKWARD ELIMINATION
# =============================================================================


import statsmodels.api as sm



X_liste = sonuc2.iloc[:, [0,1,2,3,4,5]].values

X_liste = np.array(X_liste,dtype=float)

model = sm.OLS(sonuc2.iloc[:,-1:],X_liste).fit()

print(model.summary())

#####################################
## P>|t| değerlerine göre 0.05 in üzerindeki değerler atılarak
## prediction daha iyi hale getirilmeye çalışılır

X_liste = sonuc2.iloc[:, [1,2,3,5]].values

X_liste = np.array(X_liste,dtype=float)

model = sm.OLS(sonuc2.iloc[:,-1:],X_liste).fit()

print(model.summary())


### 0,2,4 atıldı

X_liste = sonuc2.iloc[:, [1,3,5]].values

X_liste = np.array(X_liste,dtype=float)

model = sm.OLS(sonuc2.iloc[:,-1:],X_liste).fit()

print(model.summary())


X_liste = sonuc2.iloc[:, [3,5]].values

X_liste = np.array(X_liste,dtype=float)

model = sm.OLS(sonuc2.iloc[:,-1:],X_liste).fit()

print(model.summary())




regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)



#########################################################
########################################################
########## ÖNEMLİ ############

## Atılan verilere göre x_test ve x_train tekrar biçimlendirilir
# Burada sadece iki kolon kullanılarak x_train ve x_test düzenlendi

rainy_test = x_test.iloc[:, 3:4]

temp_test = x_test.iloc[:, -1:]

x_test = pd.concat([rainy_test, temp_test],axis=1)




rainy_train = x_train.iloc[:, 3:4]

temp_train = x_train.iloc[:, -1:]

x_train = pd.concat([rainy_train, temp_train], axis = 1)


#################################################################
## O değerlere sahip veriler atıldıktan sonra tekrar tahmin edilir

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)





