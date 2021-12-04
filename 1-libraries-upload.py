# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:13:01 2021

@author: zenit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veri = pd.read_csv('C:/Users/zenit/Desktop/STUDY/Python/BTK_Akademi_Machine_Learning/veriler.csv')
print(veri)
type(veri)

boy = veri[["boy"]]

boy_kilo = veri[["boy", "kilo"]]


eksikveri = pd.read_csv('C:\\Users\\zenit\\Desktop\\STUDY\\Python\\BTK_Akademi_Machine_Learning\\eksikveriler.csv')