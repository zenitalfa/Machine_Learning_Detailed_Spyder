# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:02:17 2021

@author: zenit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eksikveri = pd.read_csv('C:\\Users\\zenit\\Desktop\\STUDY\\Python\\BTK_Akademi_Machine_Learning\\eksikveriler.csv')

print(eksikveri)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values= np.nan, strategy='mean')

Yas = eksikveri.iloc[:,1:4].values

print(Yas)

imputer = imputer.fit(Yas[:,1:4])   # fit fonksiyonu, eğitmek için kullanılır, 
# 1-4 olan kolonları öğrenmesini söylüyoruz
# öğrendiği ise mean değerleri alıp , o değerleri öğrenmesidir


Yas[:,1:4] = imputer.transform(Yas[:,1:4]) 
# bu şekilde de öğrendiğini uygular ve nan değerler ortalama ile değişir

print(Yas) # 28.5 değeri NaN değerler yerine konuldu




########## TİP DÖNÜŞÜMÜ ################

# Categorical to Numerical etc



ulke = eksikveri.iloc[:, 0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder() # bu kategorik veriler kodlanarak sayısala dönüşür

ulke[:, 0] = le.fit_transform(eksikveri.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()

print(ulke)


#######3 Verilerin Birleştirilmesi ########


sonuc1 = pd.DataFrame(data = ulke, index = range(22), columns = ["fr","tr", "us"])
print(sonuc1)

sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns=["boy", "kilo","yas"])


cinsiyet = eksikveri.iloc[:, -1].values

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ["Cinsiyet"])

s = pd.concat([sonuc1, sonuc2], axis = 1)
print(s)

s2 = pd.concat([s, sonuc3], axis = 1)
print(s2)





########################## Veri KÜMESİNİN EĞİTİM VE TEST OLARAK BÖLÜNMESİ ############



from sklearn.model_selection import train_test_split # verinin satır bazlı bölünmesi

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

## x_train, x_test bağımsız değişkenleri verir, sonuç olarak nitelendirilebilir

## y_test, y_train hedefi verir

## test_size = verinin kaçta kaçının test için kullanılacağını belirtir. Burada 7 satır test, 14 satır train

## random_state ise seed generator olarak çalışır, rassal bölünme



########################## ÖZNİTELİK ÖLÇEKLEME ############## 

from sklearn.preprocessing import StandardScaler

sc = StandardScaler() # Standard Scaler'ın çalıştırılabileceği bir obje oluşturuldu

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)



## Boy kilo yas gibi veriler birbirinden tamamen farklı veriler olduğu için bir ölçekleme
## işlemine ihtiyaç vardır. O nedenle scaling yapıldı

## Birbirine göre ölçekleme yapıldı












