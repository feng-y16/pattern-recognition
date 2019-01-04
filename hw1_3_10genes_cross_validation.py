import numpy as np
import pandas as pd
import sklearn
from sklearn import discriminant_analysis
from sklearn.model_selection import train_test_split
import math
from numpy import *
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import plotall

#r=np.load("10gene_withoutnorm.npz")
r=np.load("10genes.npz")
train_data=r["arr_0"]
train_label=r["arr_1"]
train_size=r["arr_5"]
#######################################################################################################
index=range(0,10)#shuffle
train_total=shuffledata(train_data,train_label,train_size,10)
train_data=train_total[:,index]
train_label=train_total[:,10]
#######################################################################################################
times=5000
score=0
for i in range(0,times):
    model=sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=1, store_covariance=False, tol=0.0001)
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_label,test_size=0.1, random_state=i)
    model.fit(X_train, y_train)
    score=score+model.score(X_test,y_test)
    print(model.score(X_test,y_test))
print("average=",end="")
print(score/times)
#average=0.957151351351345