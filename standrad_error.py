# -*- coding: utf-8 -*-

import math

import numpy as np


#存放多个随机样本

# list_samples=[0.9922,0.9905,0.9934,0.992,0.9904,0.9903,0.9918,0.9923,0.9923,0.9912] # DMC vs Random (0.9916 0.0003)
# list_samples=[0.8236,0.8179,0.8274,0.8266,0.823,0.8196,0.8257,0.8183,0.8228,0.8268] # DMC vs Rule (0.8232 0.0011)
list_samples=[0.9518,0.951,0.952,0.9463,0.948,0.9481,0.948,0.9491,0.9491,0.9483] # Rule vs Random (0.9492 0.0006)


#求单个样本估算的标准误

def Standard_error(sample):
    
    print(np.mean(sample)) # 样本平均值

    std=np.std(sample,ddof=0)

    standard_error=std/math.sqrt(len(sample))

    return standard_error


#随机抽样

print(Standard_error(list_samples))
