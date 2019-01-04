import numpy as np
import math
from numpy import *
from collections import OrderedDict
from matplotlib import pyplot as plt
def plotline(data,label,decide,size,feature):
    plt.scatter(x=data[:,0],y=data[:,1],s=10,c=label)
    plt.show()
    return 0

def plotall(train_data,train_label,train_decide,test_data,test_label,test_decide,train_size,test_size,test_exist=False,linex=None,liney=None):
    colors = ['b','g','r','orange']
    labelpool=['E3_right','E3_wrong','E5_right','E5_wrong']
    train_decide=train_decide.astype(int)
    for i in range(0,train_size):
        if train_label[i]==0:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=9,c=colors[train_decide[i]],marker='o',label=labelpool[train_decide[i]])
        else:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=9,c=colors[train_decide[i]],marker='v',label=labelpool[3-train_decide[i]])
    if(test_exist):
        test_decide=test_decide.astype(int)
        for i in range(0,test_size):
            if test_label[i]==0:
                plt.scatter(x=test_data[i,0],y=test_data[i,1],s=9,c=colors[test_decide[i]],marker='o',label=labelpool[train_decide[i]])
            else:
                plt.scatter(x=test_data[i,0],y=test_data[i,1],s=9,c=colors[test_decide[i]],marker='v',label=labelpool[3-train_decide[i]])
    plt.plot(linex,liney,color="red")
    plt.xlabel('normalized_log(BUB1+1)')
    plt.ylabel('normalized_log(DNMT1+1)')
    handles, labels = plt.gca().get_legend_handles_labels()  
    by_label = OrderedDict(zip(labels, handles))  
    handle=[]
    for j in range(0,4):
        for i in range(0,4):
            if list(by_label.keys())[i]==labelpool[j]:
                handle.append(list(by_label.values())[i])
    plt.legend(handle, labelpool,loc = 'upper left')
    #plt.title("Perceptron")
    plt.show()
    return 0
if __name__ == '__main__':
    bayes_1=np.load("Bayes_1.npz")
    bayes_2=np.load("Bayes_2.npz")
    fisher=np.load("Fisher.npz")
    perceptron=np.load("Perceptron.npz")
    r=np.load("2genes.npz")
    train_data=r["arr_0"]
    train_label=r["arr_1"]
    test_data=r["arr_2"]
    test_label=r["arr_3"]
    label_num=r["arr_7"]
    train_size=r["arr_5"]
    test_size=r["arr_6"]
    for i in range(0,train_size):
        if train_label[i]==0:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=9,c='b',marker='o',label='E3')
        else:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=9,c='g',marker='v',label='E5')
    for i in range(0,test_size):
        if test_label[i]==0:
            plt.scatter(x=test_data[i,0],y=test_data[i,1],s=9,c='b',marker='o',label='E3')
        else:
            plt.scatter(x=test_data[i,0],y=test_data[i,1],s=9,c='g',marker='v',label='E5')
    plt.plot(bayes_1["arr_0"],bayes_1["arr_1"],label='bayes_1:1')
    plt.plot(bayes_2["arr_0"],bayes_2["arr_1"],label='bayes_1:5')
    plt.plot(fisher["arr_0"],fisher["arr_1"],label='fisher')
    plt.plot(perceptron["arr_0"],perceptron["arr_1"],label='perceptron')
    handles, labels = plt.gca().get_legend_handles_labels()  
    by_label = OrderedDict(zip(labels, handles))  
    plt.legend(by_label.values(),by_label.keys())
    plt.xlabel('normalized_log(BUB1+1)')
    plt.ylabel('normalized_log(DNMT1+1)')
    #plt.savefig("allin1.png")
    plt.show()