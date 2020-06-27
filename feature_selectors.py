import csv
import math
import sys

import numpy as np

import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression


def mylog(x):
    if x==0:
        return -10000000000
    else:
        return math.log(x)

def entropy(probs, neg, pos):
    '''
    entropy for binary classification data
    '''
    entropy=0.0
    entropy=(probs[1]/pos-probs[0]/neg)*mylog((probs[1]*neg)/(probs[0]*pos))

    return entropy

def get_bin_from_score(score):
    '''
    get bin number. 0-low, 1-high. avg. threshold=280
    '''
    return min(1,int(score//280))

def iv(header, data):
    '''
    Information Value based feature selection
    '''
    neg,pos = 0,0
    probs=np.zeros((9,5,2))

    for datum in data:
        score_bin=get_bin_from_score(float(datum[9]))
        if(score_bin==0):
            neg+=1
        else:
            pos+=1

        for i in range(9):
            probs[i][int(datum[i])-1][score_bin]+=1

    feature_score=[0 for _ in range(9)]

    for i in range(9):
        for j in range(5):
            feature_score[i]+=entropy(probs[i][j], neg, pos)

    fig = plt.figure()
    plt.barh(header[::-1],feature_score[::-1])
    plt.show()

def anova(header, regressors, target):
    '''
    ANOVA based feature selection
    '''
    # chi_scores = chi2(regressors,target)
    anova_scores = f_regression(regressors, target)

    fig = plt.figure()
    plt.barh(header[::-1],anova_scores[0][::-1])
    plt.show()

def mutual_info(header, regressors, target):
    '''
    Mutual Information based feature selection
    '''
    # chi_scores = chi2(regressors,target)
    mi_scores = mutual_info_regression(regressors, target)

    fig = plt.figure()
    plt.barh(header[::-1],mi_scores[::-1])
    plt.show()

def main():
    '''
    reads training data and calls appropriate method for feature-selection
    '''
    data,regressors,target = [],[],[]
    with open('data.csv','r') as csv_file:
        csv_reader=csv.reader(csv_file,delimiter=',')
        for row in csv_reader:
            data.append(row)
    
    header=data[0]
    data=data[1:]

    for datum in data:
        for i in range(9):
            datum[i]=int(datum[i])
        regressors.append(datum[:9])
        target.append(float(datum[9]))

    if len(sys.argv)<2:
        print('Usage: python feature_selectors.py [iv/anova/mi]')
    else:
        option=sys.argv[1]

        if option=='iv':
            iv(header, data)
        elif option=='anova':
            anova(header, regressors, target)
        elif option=='mi':
            mutual_info(header, regressors, target)
        else:
            print('Usage: python feature_selectors.py [iv/anova/mi]')

    
if __name__=='__main__':
    main()
