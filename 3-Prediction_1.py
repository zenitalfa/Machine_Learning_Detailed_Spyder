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

X_train = sc.fit_transform(x_train)
Y_train = sc.fit_transform(y_train)

## Ay ve satış verileri farklı  veriler olduğu için birbirine göre ölçeklendi


