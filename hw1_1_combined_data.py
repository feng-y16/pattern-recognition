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
label_num=r["arr_7"]
train_size=r["arr_5"]
test_size=r["arr_6"]
#######################################################################################################
datatemp=np.zeros((train_size+test_size,2))#shuffle
labeltemp=np.zeros(train_size+test_size)
datatemp[range(0,train_size),:]=train_data
datatemp[range(train_size,train_size+test_size),:]=test_data
labeltemp[range(0,train_size)]=train_label
labeltemp[range(train_size,train_size+test_size)]=test_label
total=shuffledata(datatemp,labeltemp,train_size+test_size,2)
index=range(0,2)
data=total[:,index]
label=total[:,2]
#######################################################################################################
def selectdata( data,label,aimlabel,label_num ):#选取特定标签的数据
    specificlist = np.zeros((label_num[aimlabel],2))
    j=0
    for i in range(0,train_size+test_size):
        if label[i]==aimlabel:
            specificlist[j,:]=data[i,:]
            j=j+1
    return specificlist

data1=selectdata(data,label,0,label_num)
data2=selectdata(data,label,1,label_num)
data1_mean=np.mean(data1,axis=0)
data1_cov=matrix(np.cov(data1.T))*(len(data1)-1)/len(data1)
data2_mean=np.mean(data2,axis=0)
data2_cov=matrix(np.cov(data2.T))*(len(data2)-1)/len(data2)
inverse1=data1_cov.I
inverse2=data2_cov.I
print("E3_mean=",end="")
print(data1_mean)
print("E3_cov=",end="")
print(inverse1)
print("E5_mean=",end="")
print(data2_mean)
print("E5_cov=",end="")
print(inverse2)
#######################################################################################################
P=[0.5,0.5]#这里需要去掉下一行的注释符号以获得另一个先验概率的结果，如果要输出还需去掉np.savez("Bayes_1",linex,liney)的注释符号
#P=[1.0/6,5.0/6]#如果要输出需去掉np.savez("Bayes_2",linex,liney)的注释符号
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
for i in range(0,train_size+test_size):
    if(decidefunction(data[i,:])==label[i]):
        true=true+1
    else:
        false=false+1
print("accuracy=",end="")
print(true/(true+false))
decide=np.zeros(train_size+test_size)
for i in range(0,train_size+test_size):
    decide[i]=decidefunction(data[i,:])
#######################################################################################################
x_begin=min(data[:,0])#扫描分界面位置
x_end=max(data[:,0])
y_begin=min(data[:,1])-1
y_end=max(data[:,1])+1
pointnumber=1200
linex=np.linspace(x_begin,x_end,pointnumber)
liney=np.zeros(pointnumber)
lineytest=np.linspace(y_begin,y_end,pointnumber)
def coutline():
    for i in range(0,pointnumber):
        for j in range(pointnumber-1,0,-1):
            if(decidefunction([linex[i],lineytest[j]])+decidefunction([linex[i],lineytest[j-1]])==1):
                break
        liney[i]=lineytest[j]
        #print(i)
    return liney
coutline()
#######################################################################################################
#输出数据
#np.savez("Bayes_1",linex,liney)
#np.savez("Bayes_2",linex,liney)
plotall(data,label,decide,None,None,None,train_size+test_size,0,False,linex,liney)
#withoutnorm
#P1:0.9890829694323144
#P2:0.9890829694323144
#withnorm
#P1:0.9890829694323144
#P2:0.9890829694323144