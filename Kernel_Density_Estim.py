# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:27:59 2020

@author: Abdullah Hamza Şahin
"""

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity


data = load_iris()
X = data.data[:,2:]
Y = data.target
Y = Y.reshape((len(Y),1))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=139)

X_train0 = np.zeros((1,2))
X_train1 = np.zeros((1,2))
X_train2 = np.zeros((1,2))

for i in range(len(y_train)):
    if y_train[i] == 0:
        X_train0 = np.concatenate((X_train0,X_train[i].reshape(1,2)), axis=0)
    elif y_train[i] == 1:
        X_train1 = np.concatenate((X_train1,X_train[i].reshape(1,2)),axis=0)
    elif y_train[i] == 2:
        X_train2 = np.concatenate((X_train2,X_train[i].reshape(1,2)),axis=0)

X_train0 =  X_train0[1:,:]
X_train1 =  X_train1[1:,:]
X_train2 =  X_train2[1:,:]


errors_gaussian = []
bandwidth = np.linspace(0.1,2,4)
for b in bandwidth:
    model = KernelDensity(bandwidth=b, kernel="gaussian")
    
    m0 = model.fit(X_train0)
    s0 = m0.score_samples(X_test).reshape(len(X_test),1)
    
    m1 = model.fit(X_train1)
    s1 = m1.score_samples(X_test).reshape(len(X_test),1)
    
    m2 = model.fit(X_train2)
    s2 = m2.score_samples(X_test).reshape(len(X_test),1)
   
    scores = np.hstack((s0,s1,s2))
    preds = np.empty((len(scores),1))
    counter = 0
    for l in scores:
        mx = np.argmax(l)
        if mx == 0:
            preds[counter] = 0
        elif mx == 1:
            preds[counter] = 1
        elif mx == 2:
            preds[counter] = 2
        counter += 1
    
    error = np.size(y_test == preds) - np.count_nonzero(y_test == preds)
    errors_gaussian.append(error)

ind = np.argmin(errors_gaussian)
print("The optimal bandwidth for the gaussian kernel is ",bandwidth[ind])
print(errors_gaussian[ind]," samples wrong classified with the optimal bandwidth.")
    
errors_tophat = []
for b in bandwidth:
    model = KernelDensity(bandwidth=b, kernel="tophat")
    
    m0 = model.fit(X_train0)
    s0 = m0.score_samples(X_test).reshape(len(X_test),1)
    
    m1 = model.fit(X_train1)
    s1 = m1.score_samples(X_test).reshape(len(X_test),1)
    
    m2 = model.fit(X_train2)
    s2 = m2.score_samples(X_test).reshape(len(X_test),1)
   
    scores = np.hstack((s0,s1,s2))
    preds = np.empty((len(scores),1))
    counter = 0
    for l in scores:
        mx = np.argmax(l)
        if mx == 0:
            preds[counter] = 0
        elif mx == 1:
            preds[counter] = 1
        elif mx == 2:
            preds[counter] = 2
        counter += 1
    
    error = np.size(y_test == preds) - np.count_nonzero(y_test == preds)
    errors_tophat.append(error)

ind = np.argmin(errors_tophat)
print("The optimal bandwidth for the tophat kernel is ",bandwidth[ind])
print(errors_tophat[ind]," samples wrong classified with the optimal bandwidth.")

