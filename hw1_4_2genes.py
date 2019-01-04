import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
times=5000
score=zeros(2)
for i in range(0,times):
    model=sklearn.linear_model.SGDClassifier(loss="perceptron",tol=1e-3, eta0=1, learning_rate="invscaling", penalty=None)
    model.fit(train_data,train_label)
    print("train accuracy=",end="")
    print(accuracy_score(train_label,model.predict(train_data)))
    score[0]=score[0]+accuracy_score(train_label,model.predict(train_data))
    print("test accuracy=",end="")
    print(accuracy_score(test_label,model.predict(test_data)))
    score[1]=score[1]+accuracy_score(test_label,model.predict(test_data))
print("average train accuracy=",end="")
print(score[0]/times)
print("average test accuracy=",end="")
print(score[1]/times)
#withoutnorm
#average train accuracy=0.9613389645776399
#average test accuracy=0.9527230769230846
#withnorm
#average train accuracy=0.9812114441416885
#average test accuracy=0.9893670329671087