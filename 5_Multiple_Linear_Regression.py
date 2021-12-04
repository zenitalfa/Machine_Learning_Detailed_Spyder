# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 19:06:16 2021

@author: zenit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriSeti = pd.read_csv("veriler.csv")
print(veriSeti)


boy = veriSeti[["boy"]]
print(boy)

kilo = veriSeti[["kilo"]]
print(kilo)

yas = veriSeti[["yas"]]
print(yas)

cinsiyet = veriSeti[["cinsiyet"]]
print(cinsiyet)



################################## PREPROCESSING #######################

## Categorical to Array

## Encoder


## ulke için encoding

ulke = veriSeti.iloc[:, 0:1].values
print(ulke)
type(ulke)

from sklearn import preprocessing # class oluşturuldu

le = preprocessing.LabelEncoder()  # obje oluşturuldu

ulke[:, 0] = le.fit_transform(veriSeti.iloc[: ,0])

print(ulke)

ohe = preprocessing.OneHotEncoder() # Encoder için obje oluşturuldu

ulke = ohe.fit_transform(ulke).toarray()

print(ulke)



## Cinsiyet için encoding

cins = veriSeti.iloc[:, -1:].values
print(cins)
type(cins)

from sklearn import preprocessing # class oluşturuldu

le = preprocessing.LabelEncoder()  # obje oluşturuldu

cins[:, -1] = le.fit_transform(veriSeti.iloc[: ,-1])

print(ulke)

ohe = preprocessing.OneHotEncoder() # Encoder için obje oluşturuldu

cins = ohe.fit_transform(cins).toarray()

print(cins)


##############################################################################
#################################### VERİLERİN BİRLEŞTİRİLMESİ ###########
############### pd.concat ###########
#################################################################################

"""
 Dummy variable trap tan kaçınmak için cins verisinin(kodlanmış) sadece tek kolonu alınmalıdır
 Çünkü 0 ve 1 birbirini tamamlayan ilinitili değişkenlerdir
"""


sonuc = pd.DataFrame(data = ulke, index = range(22), columns=["fr", "tr","us"])

Yas = veriSeti.iloc[:,1:4].values

sonuc2 = pd.DataFrame(data = Yas , index = range(22) , columns=["boy", "kilo","yas"])

print(sonuc2)

sonuc3 = pd.DataFrame(data = cins[:, :1], index = range(22), columns=["cinsiyet"])

print(sonuc3)



s= pd.concat([sonuc, sonuc2], axis= 1)

print(s)


s2 = pd.concat([s,sonuc3], axis = 1)

print(s2)




##############################################################
######################## Verilerin ayrılması ###################
##############################################################

### x_train, x_test_ y_train, y_test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3, test_size=0.33, random_state= 0)

################# Ölçekleme #####################

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  # obje oluşturuldu

# veriler birbirine x_train ve x_test'e göre ölçeklendi

X_train = sc.fit_transform(x_train) 
X_test = sc.fit_transform(x_test)







##############################################################################
############################ Multiple Linear Regression ######################
######################################################################

################### Model İnşası ##################################
#################################################


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()  # obje oluşturuldu


regressor.fit(x_train, y_train)

""" ÖNEMLİİİİİİİİİ
BURADA "regressor" objesinin yaptığı iş : 
    
    x_train'den (bağımsız değişkenlerinden) y_train(bağımlı değişkenini) öğrenmektir
    
    yani aralarında LİNEER BİR MODEL KURMAK anlamına gelir
    
    """
    
    ## x_train ile makine eğitildi
    ## x_test ile bir tahmin algoritması çıkarılacaktır
    
    
y_pred = regressor.predict(x_test)




#########################################################################
############################## BOY kolonuna göre MULTİPLE REGRESSİON ##################
######################################################################


print(s2)

boy = s2.iloc[:,3:4]
print(boy)

sol = s2.iloc[:,:3]
print(sol)

sag = s2.iloc[:,4:]
print(sag)

### Boy kolonu verinin içinden alınarak geri kalan sol ve sağ sütunlar birleştirildi
## pd.concat([sag,sol],axis = 1)


s3 = pd.concat([sol,sag], axis = 1)
print(s3)


#######################################################
##################### Eğitim ve test verilerinin hazırlanması ##############################


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s3, boy, test_size=0.33, random_state = 0)


#######################################################
####################### ÖLÇEKLEME #####################################


# =============================================================================
# from sklearn.preprocessing import StandardScaler
# 
# sc = StandardScaler() # obje oluşturuldu
# 
# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)
# 
# print(X_train)
# print(X_test)
# =============================================================================


##############################################################
######################## Model İnşası/ Regression ##################


from sklearn.linear_model import LinearRegression


regression1 = LinearRegression()


regression1.fit(x_train,y_train)


y_pred_boy = regression1.predict(x_test) # bu tahmin ile y_test arasındaki farka bakılabilir


###########################################
################### iSTATİSTİKİ ANALİZ ############
############### MODEL BAŞARIMI ###################













