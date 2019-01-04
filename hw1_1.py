import numpy as np
import pandas as pd
import sklearn
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
def selectdata( train_data,train_label,label,label_num ):#选取特定标签的数据
    specificlist = np.zeros((label_num[label],2))
    j=0
    for i in range(0,train_size):
        if train_label[i]==label:
            specificlist[j,:]=train_data[i,:]
            j=j+1
    return specificlist

data1=selectdata(train_data,train_label,0,label_num)
data2=selectdata(train_data,train_label,1,label_num)
data1_mean=np.mean(data1,axis=0)
data1_cov=matrix(np.cov(data1.T))*(len(data1)-1)/len(data1)
data2_mean=np.mean(data2,axis=0)
data2_cov=matrix(np.cov(data2.T))*(len(data2)-1)/len(data2)
inverse1=data1_cov.I
inverse2=data2_cov.I
print("E3_mean=",end="")
print(data1_mean)
print("E3_cov_inverse=",end="")
print(inverse1)
print("E5_mean=",end="")
print(data2_mean)
print("E5_cov_inverse=",end="")
print(inverse2)
#######################################################################################################
P=[0.5,0.5]#下面会更改P值重算一遍，共用上面代码的结果
print("P=",end="")
print(P)
#print(-0.5*math.log(abs(np.linalg.det(data1_cov))))
#print(-0.5*math.log(abs(np.linalg.det(data2_cov))))
b1=-0.5*math.log(abs(np.linalg.det(data1_cov)))+math.log(P[0])
b2=-0.5*math.log(abs(np.linalg.det(data2_cov)))+math.log(P[1])

def decidefunction(x):#判别函数
    g1=-0.5*np.dot(np.dot((x-data1_mean).T,inverse1),(x-data1_mean))+b1
    g2=-0.5*np.dot(np.dot((x-data2_mean).T,inverse2),(x-data2_mean))+b2
    if g1>g2:
        return 0
    else:
        return 1
true=0
false=0
for i in range(0,train_size):
    if(decidefunction(train_data[i,:])==train_label[i]):
        true=true+1
    else:
        false=false+1
print("train accuracy=",end="")
print(true/(true+false))

true=0
false=0
for i in range(0,test_size):
    if(decidefunction(test_data[i,:])==test_label[i]):
        true=true+1
    else:
        false=false+1
print("test accuracy=",end="")
print(true/(true+false))
#######################################################################################################
P=[1.0/6,5.0/6]
print("P=",end="")
print(P)
#print(-0.5*math.log(abs(np.linalg.det(data1_cov))))
#print(-0.5*math.log(abs(np.linalg.det(data2_cov))))
b1=-0.5*math.log(abs(np.linalg.det(data1_cov)))+math.log(P[0])
b2=-0.5*math.log(abs(np.linalg.det(data2_cov)))+math.log(P[1])
true=0
false=0
for i in range(0,train_size):
    if(decidefunction(train_data[i,:])==train_label[i]):
        true=true+1
    else:
        false=false+1
print("train accuracy=",end="")
print(true/(true+false))

true=0
false=0
for i in range(0,test_size):
    if(decidefunction(test_data[i,:])==test_label[i]):
        true=true+1
    else:
        false=false+1
print("test accuracy=",end="")
print(true/(true+false))

#withoutnorm
#P1:train accuracy=0.989100817438692,test accuracy=1.0
#P2:train accuracy=0.9863760217983651,test accuracy=1.0
#withnorm
#P1:train accuracy=0.989100817438692,test accuracy=1.0
#P2:train accuracy=0.9863760217983651,test accuracy=1.0