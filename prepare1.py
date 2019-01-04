import numpy as np
import pandas as pd
import sklearn
import math
from numpy import *
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import *
from sklearn import preprocessing
offset=1
train_original_data=pd.read_csv("F:\\pythonhomework\\hw1\\E3E5_data\\train_data_E3E5_2genes.txt",sep=' ')
train_size=len(train_original_data["BUB1"])
train_data=np.zeros((train_size,2))
train_label=np.zeros(train_size)
label_num=[0,0]
label_num_total=[0,0]

for i in range(0,train_size):
    train_data[i,0]=math.log(train_original_data.ix[i]["BUB1"]+offset)
    train_data[i,1]=math.log(train_original_data.ix[i]["DNMT1"]+offset)
    if train_original_data.ix[i]["label"]=="E3":
        train_label[i]=0
        label_num[0]=label_num[0]+1
        label_num_total[0]=label_num_total[0]+1
    else:
        train_label[i]=1
        label_num[1]=label_num[1]+1
        label_num_total[1]=label_num_total[1]+1

test_original_data=pd.read_csv("F:\\pythonhomework\\hw1\\E3E5_data\\test_data_E3E5_2genes.txt",sep=' ')
test_size=len(test_original_data["BUB1"])
test_data=np.zeros((test_size,2))
test_label=np.zeros(test_size)
#print(test_size)
for i in range(0,test_size):
    test_data[i,0]=math.log(test_original_data.ix[i]["BUB1"]+offset)
    test_data[i,1]=math.log(test_original_data.ix[i]["DNMT1"]+offset)
    if test_original_data.ix[i]["label"]=="E3":
        test_label[i]=0
        label_num_total[0]=label_num_total[0]+1
    else:
        test_label[i]=1
        label_num_total[1]=label_num_total[1]+1

np.savez("2genes_withoutnorm.npz",train_data,train_label,test_data,test_label,label_num,train_size,test_size,label_num_total)
train_range=range(0,train_size)
test_range=range(train_size,train_size+test_size)
data=np.zeros((train_size+test_size,2))
data[train_range,:]=train_data
data[test_range,:]=test_data
for i in range(0,2):
    data[:,i]=sklearn.preprocessing.scale(data[:,i])
train_data=data[train_range,:]
test_data=data[test_range,:]
np.savez("2genes.npz",train_data,train_label,test_data,test_label,label_num,train_size,test_size,label_num_total)