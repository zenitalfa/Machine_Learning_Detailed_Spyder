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

