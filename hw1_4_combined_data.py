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
label_num=r["arr_7"]
train_size=r["arr_5"]
test_size=r["arr_6"]
#######################################################################################################
index=range(0,2)#shuffle
datatemp=np.zeros((train_size+test_size,2))
labeltemp=np.zeros(train_size+test_size)
datatemp[range(0,train_size),:]=train_data
datatemp[range(train_size,train_size+test_size),:]=test_data
labeltemp[range(0,train_size)]=train_label
labeltemp[range(train_size,train_size+test_size)]=test_label
total=shuffledata(datatemp,labeltemp,train_size+test_size,2)
data=total[:,index]
label=total[:,2]
#######################################################################################################
times=5000
score=0
score_list=[]
for i in range(0,times):
    model_test=sklearn.linear_model.SGDClassifier(loss="perceptron", eta0=1, tol=1e-3,learning_rate="invscaling", penalty=None)
    model_test.fit(data,label)
    score=score+accuracy_score(label,model_test.predict(data))
    score_list.append(accuracy_score(label,model_test.predict(data)))
print("average accuracy=",end="")
print(score/times)
print("maximum accuracy=",end="")
print(max(score_list))
print("minimum accuracy=",end="")
print(min(score_list))
#######################################################################################################
model=sklearn.linear_model.SGDClassifier(loss="perceptron", eta0=1, tol=1e-3,learning_rate="invscaling", penalty=None)
model.fit(data,label)
decide=model.predict(data)
#######################################################################################################
x_begin=min(data[:,0])#扫描分界面位置
x_end=max(data[:,0])
y_begin=min(data[:,1])
y_end=max(data[:,1])+1
pointnumber=1200
linex=np.linspace(x_begin,x_end,pointnumber)
liney=np.zeros(pointnumber)
linextest=np.ones(pointnumber)
tempdata=np.zeros((pointnumber,2))
tempdata[:,1]=np.linspace(y_begin,y_end,pointnumber)
def coutline():
    for i in range(0,pointnumber):
        tempdata[:,0]=linex[i]*linextest
        temp=model.predict(tempdata)
        for j in range(1,pointnumber):
            if(temp[j]+temp[j-1]==1):
                break
        liney[i]=tempdata[j][1]
    return liney
coutline()
plotall(data,label,decide,None,None,None,train_size+test_size,0,False,linex,liney)
#np.savez("Perceptron",linex,liney)
#withoutnorm
#average accuracy=0.9628288209606433
#withnorm
#average accuracy=0.983187772925717
