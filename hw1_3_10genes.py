import numpy as np
import pandas as pd
import sklearn
from sklearn import discriminant_analysis
import math
from numpy import *
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import plotall

#r=np.load("10gene_withoutnorm.npz")
r=np.load("10genes.npz")
train_data=r["arr_0"]
train_label=r["arr_1"]
test_data=r["arr_2"]
test_label=r["arr_3"]
label_num=r["arr_4"]
train_size=r["arr_5"]
test_size=r["arr_6"]
#######################################################################################################
index=range(0,10)#shuffle
train_total=shuffledata(train_data,train_label,train_size,10)
train_data=train_total[:,index]
train_label=train_total[:,10]
#######################################################################################################
model=sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=1, store_covariance=False, tol=0.0001)
model.fit(train_data,train_label)

#temp=np.zeros(10)
#temp=model.coef_
#coef=np.zeros(10)
#for i in range(0,10):
#    coef[i]=temp[0,i]

print("train accuracy=",end="")
print(model.score(train_data,train_label))
print("test accuracy=",end="")
print(model.score(test_data,test_label))
#withoutnorm
#train accuracy=0.9673024523160763
#test accuracy=0.978021978021978
#withnorm
#train accuracy=0.9673024523160763
#test accuracy=0.978021978021978