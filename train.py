import csv
import math
import sys 
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt
import tensorflow as tf

def matching_bins(x_list, y_list):
    '''
    returns the fraction of matching elements from the 2 lists. 
    '''
    acc, tot=0, 0
    for x,y in zip(x_list, y_list):
        tot+=1
        if x==y:
            acc+=1
    
    return acc/tot

def bin_metric(x_list,y_list,k):
    '''
    returns the fraction of elements belonging to the same bin,
    when continuous variable discretized into k bins.
    '''
    cor, tot=0,0
    min_score, max_score=min(y_list), max(y_list)
    ratio=(max_score-min_score)//k
    for x,y in zip(x_list, y_list):
        tot+=1
        
        if min(ratio-1,(x-min_score)//ratio)==min(ratio-1,(y-min_score)//ratio):
            cor+=1

    return cor/tot

def encode(x):
    '''
    one-hot encoding of length 5, for value=x (1<=x<=5)
    '''
    l=[0 for _ in range(5)]
    l[x-1]=1
    return l

def generate_woe(data, label):
    '''
    generate weight of evidence for each category of each feature, 
    threshold for binary classification = 275
    '''
    pos,neg=0,0
    woe=[[[0.0,0.0] for _ in range(5)] for _ in range(7)]

    for i in range(len(data)):
        for j,val in enumerate(data[i]):
            if label[i]>275:
                woe[j][val-1][0]+=1
                pos+=1
            else:
                woe[j][val-1][1]+=1
                neg+=1

    for i in range(7):
        for j in range(5):
            if(woe[i][j][0]==0 and woe[i][j][1]==0):
                woe[i][j]=0
            elif(woe[i][j][1]==0):
                woe[i][j]=1000
            elif(woe[i][j][0]==0):
                woe[i][j]=-1000
            else:
                woe[i][j]=math.log((woe[i][j][0]*neg)/(woe[i][j][1]*pos))*100

    return woe 


def processed_data(data):
    '''
    returns the one-hot encoded feature set and target labels from the data. (prepeocessing step)
    '''
    one_hot_data_list=[[] for _ in range(len(data))]
    target=[0.0 for _ in range(len(data))]

    for i in range(len(data)):
        one_hot_data=[]
        for val in data[i][:7]:
            one_hot_data.extend(encode(int(val)))
        one_hot_data_list[i]=one_hot_data
        target[i]=int(float((data[i][-1])))

    return one_hot_data_list, target


def woe_reg(train_data, train_label, test_data, test_label):
    '''
    Weight of Evidence (WoE) based model
    '''
    train_data=np.array(train_data)[:,:-3]
    test_data=np.array(test_data)[:,:-3]
    
    woe=generate_woe(train_data, train_label)

    train_score=train_label
    test_score=test_label

    woe_train_data=[[0.0 for _ in range(7)] for _ in range(len(train_data))]
    woe_test_data=[[0.0 for _ in range(7)] for _ in range(len(test_data))]

    for i in range(len(train_data)):
        for j in range(7):
            woe_train_data[i][j]=woe[j][train_data[i][j]-1]

    for i in range(len(test_data)):
        for j in range(7):
            woe_test_data[i][j]=woe[j][test_data[i][j]-1]

    test_data=np.array(test_data)

    reg = Ridge(alpha=0.1).fit(woe_train_data, train_score)

    predicted_scores=reg.predict(woe_test_data)

    print("SSE= "+str(mean_squared_error(predicted_scores,test_score)))

    for i in range(8):
        acc=bin_metric(predicted_scores, test_score, i+1)
        print("number of bins= "+str(i+1)+": "+str(acc*100)+'%')

    fig = plt.figure()
    plt.plot(np.arange(400), test_score,'b')
    plt.plot(np.arange(400), predicted_scores,'r')
    plt.show()

def linreg(one_hot_train_data, train_label, one_hot_test_data, test_label):
    '''
    Linear Ridge regression model 
    '''
    reg = Ridge(alpha=0.1).fit(one_hot_train_data, train_label)

    # print(reg.coef_)
    # print(reg.intercept_)
    predicted_labels=reg.predict(one_hot_test_data)

    print("SSE= "+str(mean_squared_error(predicted_labels,test_label)))

    for i in range(8):
        acc=bin_metric(predicted_labels, test_label, i+1)
        print("number of bins= "+str(i+1)+": "+str(acc*100)+'%')

    fig = plt.figure()
    plt.plot(np.arange(400), test_label,'b')
    plt.plot(np.arange(400), predicted_labels,'r')
    plt.show()

def nn(one_hot_train_data, train_label, one_hot_test_data, test_label):
    '''
    Neural Network based model
    '''
    one_hot_train_data=np.array(one_hot_train_data)
    train_label=np.array(train_label)
    one_hot_test_data=np.array(one_hot_test_data)

    inputs=tf.keras.layers.Input(shape=(35,))
    hidden1=tf.keras.layers.Dense(15,name='hidden1', activation=tf.nn.relu)(inputs)
    output=tf.keras.layers.Dense(1,name='output', activation=tf.nn.relu)(hidden1)

    model=tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss="mse", metrics=["mse","mae"])
    model.fit(one_hot_train_data, train_label, batch_size=128, epochs=100, validation_split=0.2)

    predicted_labels=model.predict(one_hot_test_data)
    
    print("SSE= "+str(mean_squared_error(predicted_labels,test_label)))

    for i in range(8):
        acc=bin_metric(predicted_labels, test_label, i+1)
        print("number of bins= "+str(i+1)+": "+str(acc*100)+'%')

    fig = plt.figure()
    plt.plot(np.arange(400), test_label,'b')
    plt.plot(np.arange(400), predicted_labels,'r')
    plt.show()

def main():
    '''
    calls the appropriate method based on user-input for model training
    '''
    train_data, test_data=[],[]
    with open('data.csv','r') as csv_file:
        csv_reader=csv.reader(csv_file,delimiter=',')
        for i,row in enumerate(csv_reader):
            if(i<=1600):
                train_data.append(row)
            else:
                test_data.append(row)

    header=train_data[0]
    train_data=train_data[1:]

    for datum in train_data:
        for i in range(9):
            datum[i]=int(datum[i])
        datum[9]=int(float(datum[9]))

    for datum in test_data:
        for i in range(9):
            datum[i]=int(datum[i])
        datum[9]=int(float(datum[9]))

    one_hot_train_data, train_label=processed_data(train_data)
    one_hot_test_data, test_label=processed_data(test_data)

    if len(sys.argv)<2:
        print('Usage: python train.py [woe/linreg/nn]')
    else:
        option=sys.argv[1]

        if option=='woe':
            woe_reg(train_data, train_label, test_data, test_label)
        elif option=='linreg':
            linreg(one_hot_train_data, train_label, one_hot_test_data, test_label)
        elif option=='nn':
            nn(one_hot_train_data, train_label, one_hot_test_data, test_label)
        else:
            print('Usage: python feature_selectors.py [iv/anova/mi]')


if __name__=='__main__':
    main()