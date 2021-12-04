# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 19:59:00 2021

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
 Dummy variable trap tan kaçınmak için cins verisinin(kodlanmış) sadece tejk kolonu alınmalıdır
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







###################################################################################
############################### GERİ ELEME YÖNTEMİ(BACKWARD ELİMİNATİON) ############################

import statsmodels.api as sm


# =============================================================================
#  Burada sabit bir BETA0 değeri oluşturmamız gerekiyor. Multi regression modeli için
#   y = Beta0 + Beta1. x1 + Beta2 . x2 ...... +e
# Bu Beta0 değeri bir nevi karar noktası gibi davranabilir
# böylece hangi değişkenlerin makine öğrenmesi üzerinde etkili olduğu anlaşılarak
# p < SL koşulunu sağlayan değişkenler sistem içinde tutulur
# Diğerleri elemine edilir
# =============================================================================

X= np.append(arr= np.ones((22,1)).astype(int), values= s2, axis = 1) 


# =============================================================================
#  np.append yaparak 1'lerden oluşan 22,1 diziyi oluşturarak s2 dataFrame'inden gelen
#   gelen verilerle birleştirerek bir array oluşturdu. Axis = 1 olması dikey olarak ekleyeceğini
#   ifade etmektedir.
# =============================================================================

X_liste = s3.iloc[:, [0,1,2,3,4,5]].values

## Bu X_listeden eleme yaparak gideceğiz, geriye eliminasyon

# =============================================================================
# Aşağıdaki işlem X_listenin np arraye dönüştürülmesi ve
# ÖNEMLİİİİİİİİİİİİİİİİİ
# Daha sonra da X_liste içerisindeki bağımsız değişkenlerin boy ile ilişkisinin
# bulunduğu bir model oluşturulması üzerinedir.
# =============================================================================


X_liste = np.array(X_liste,dtype=float)

model = sm.OLS(boy,X_liste).fit()

print(model.summary())

# =============================================================================
#  model print edildiğinde özette verilen x1,x2,x3,x4,x5,x6 değerlerinden
# P>(t) değeri 0.05 (SL) üzerinde olanları çıkartmak gerekir
# bunun için listeden o değişken çıkartılır
# =============================================================================


X_liste = s3.iloc[:, [0,1,2,3,5]].values

X_liste = np.array(X_liste,dtype=float)

model = sm.OLS(boy,X_liste).fit()

print(model.summary())




X_liste = s3.iloc[:, [0,1,2,3]].values

X_liste = np.array(X_liste,dtype=float)

model = sm.OLS(boy,X_liste).fit()

print(model.summary())




















