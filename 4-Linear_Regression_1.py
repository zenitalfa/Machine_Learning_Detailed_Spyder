# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:41:14 2021

@author: zenit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


########### Veriler (dataframe) ayrıştırıldı ###############

veriSeti = pd.read_csv('satislar.csv')

print(veriSeti)
print(type(veriSeti))

aylar = veriSeti[["Aylar"]]
print(aylar)

satislar = veriSeti[["Satislar"]]
print(satislar) 

"""
iloc ile de bölünebilir

aylar = veriSeti.iloc[:,0].values
satislar = veriSeti.iloc[:,-1].values
"""


################ Alınan veriler test ve eğitim için bölündü ###########################
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar, test_size= 0.33, random_state= 0)






################### Verileri Ölçekleme ###########################


from sklearn.preprocessing import StandardScaler

sc = StandardScaler() # sc artık ölçekleme işlevi yapacaktır


## Ay ve satış verileri farklı  veriler olduğu için birbirine göre ölçeklendi

X_train = sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)

print(X_train)
print(X_test)


Y_train = sc.fit_transform(y_train)
Y_test= sc.fit_transform(y_test)

print(Y_train)
print(Y_test)



#################################### Simple Linear Regression  #################

##### Model İnşası

## CTRL+I tuşuna basarak bu sınıfların üzerine mouse ile gelip bilgi alınabilir

from sklearn.linear_model import LinearRegression # Linear regression class'ı çağırıldı

lr = LinearRegression() # bir obje oluşturduk

## lr.fit() dersek modeli inşa etmeye başlarız

## model kendini eğitir 

lr.fit(x_train, y_train) 

## Tahmin için lr.predict

tahmin = lr.predict(x_test)

## Burada tahmin, Y_test 'e yakın değerleri X_test ile elde etmemizdir
## , karşılaştırma yaparak eğitimideğerlendirebiliriz


""" Bu model inşasında şimdilik ham veri kullanıldı. Yani sadece x_train, y_train. 

     Ölçeklenmiş veriler de kullanılabilirdi istenirse karıştırma
     """


######################## Görselleştirme #######################

# Elde ettiğimiz verileri görselleştirerek daha anlaşılır hale getirebiliriz

# Bu verileri sort ederek indexleri sıralı hale getiririz.
# ilk başta indexlerin sıralı olmadığına bakabilirsin

x_train = x_train.sort_index()

y_train = y_train.sort_index()

plt.plot(x_train,y_train)


## ham x_test ve train edilen x_test arasındaki kıyaslama için

plt.plot(x_test, tahmin)


plt.title("x_test vs Tahmin")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")


## Doğrusal regression bu şekildedir.