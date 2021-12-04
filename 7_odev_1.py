# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 22:30:18 2021

@author: zenit
"""

# =============================================================================
# ÖDEV-1- Regresyon
# =============================================================================

## Veri setinin içeri aktarılması 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriseti = pd.read_csv("odev_tenis.csv")

print(veriseti)

outlook = veriseti[["outlook"]]
print(outlook)


temperature = veriseti[["temperature"]]
print(temperature)

humidity = veriseti[["humidity"]]
print(humidity)

windy = veriseti[["windy"]]
print(windy)

play = veriseti[["play"]]
print(play)



# =============================================================================
# VERİLERİN HAZIRLANMASI
# =============================================================================



from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

## Play ve Windy kolonları için LABEL ENCODİNG
## Outlook kolonu için ise ONEHOTENCODİNG yapmamız gerekir


play = veriseti.iloc[: , -1:].values

print(play)

windy = veriseti.iloc[: ,-2:-1:].values

print(windy)

## Label Encoding play ve windy

play = le.fit_transform(veriseti.iloc[:,-1])


windy = le.fit_transform(veriseti.iloc[:,-2:-1])


## Outlook için Onehotencoding

outlook = veriseti.iloc[:,0].values

print(outlook)

outlook = ohe.fit_transform(outlook).toarray()




                              










