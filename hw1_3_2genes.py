import numpy as np
import pandas as pd
import sklearn
from sklearn import discriminant_analysis
import math
from numpy import *
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import plotall

#r=np.load("2genes_withoutnorm.npz")
r=np.load("2genes.npz")
train_data=r["arr_0"]
train_label=r["arr_1"]
test_data=r["arr_2"]
test_label=r["arr_3"]
label_num=r["arr_4"]
train_size=r["arr_5"]
test_size=r["arr_6"]
#######################################################################################################
index=range(0,2)#shuffle
train_total=shuffledata(train_data,train_label,train_size,2)
train_data=train_total[:,index]
train_label=train_total[:,2]
#######################################################################################################
model=sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=1, store_covariance=False, tol=0.0001)
model.fit(train_data,train_label)

#temp=np.zeros(2)
#temp=model.coef_
#coef=np.zeros(2)
#coef[0]=temp[0,0]
#coef[1]=temp[0,1]

print("train accuracy=",end="")
print(model.score(train_data,train_label))
print("test accuracy=",end="")
print(model.score(test_data,test_label))
#withoutnorm
#train accuracy=0.9863760217983651
#test accuracy=0.978021978021978
#withnorm
#train accuracy=0.9863760217983651
#test accuracy=0.978021978021978