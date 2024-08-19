#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import sys
import sympy as sp
import math
import os
import copy
import random
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
import sklearn.metrics as metrics

totaldf = pd.DataFrame(columns=['puf','algo','size','run','crp','acc'])
def record(totaldf, puf, algo, size, run, crp, acc):
    tempdf = pd.DataFrame({
    'puf':puf,
    'algo':algo,
    'size':input_bit_up,
    'run': i,
    'crp': a,
    'acc': b })
    new_totaldf = pd.concat([totaldf, tempdf],ignore_index=True)
    return new_totaldf


# In[2]:


def ANN(input_bit_up, input_bit_down, FV, RES):
    x = tf.placeholder(tf.float32,[None,max(input_bit_up,input_bit_down)])
    y = tf.placeholder(tf.float32,[None,1])

    weights = {
        'delay_up':tf.Variable(tf.random_normal([input_bit_up,1])),#tf.zeros   tf.random_normal
        'delay_down':tf.Variable(tf.random_normal([input_bit_down+1,1]))
    }

    def neural_network(x):
        delay1=tf.matmul(x[:,0:input_bit_up],weights['delay_up'])
        res_up=tf.sigmoid(delay1)
        delay2=tf.matmul(x[:,0:input_bit_down],weights['delay_down'][0:input_bit_down])+tf.multiply(res_up,weights['delay_down'][input_bit_down,0])
        res_down=tf.sigmoid(delay2)
        return res_down

    result = neural_network(x)
    prediction = tf.sign(2*result-1)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - result), reduction_indices = [1]))
    train_step = tf.train.AdamOptimizer(0.005).minimize(loss)

    correct_pred = tf.equal(prediction, 2 * y - 1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    step_num = 2000
    batch_num = 1
    acc_best = 0.0
    last_acc = []

    with tf.Session() as sess:
        for for_train in range(1,gen):
            train_size = 1/(2**for_train)
            test_size = 1 - train_size
            x_train, x_test, y_train, y_test = ms.train_test_split(FV, RES, train_size = train_size, test_size = test_size)
            batch_size = x_train.shape[0]//batch_num
            sess.run(init)
            acc = sess.run(accuracy, feed_dict = {x:x_test,y:y_test})
            for step in range(step_num + 1):
                for batch in range(batch_num):
                    batch_x, batch_y = x_train[batch_size*batch:batch_size*(batch + 1), : ], y_train[batch_size*batch:batch_size*(batch + 1), : ]
                    sess.run(train_step, feed_dict = {x:batch_x,y:batch_y})
                acc = sess.run(accuracy,feed_dict = {x:x_test, y:y_test})
                acc_train = sess.run(accuracy,feed_dict = {x:x_train, y:y_train})
                loss_ = sess.run(loss,feed_dict = {x:x_train,y:y_train})
                if step%100 == 0:
                    print("Step " + str(step) +" loss:"+str(loss_)+ " train Accuracy："  + str(acc_train) + " test accurary:"+str(acc))
            last_acc.append(acc)
    print(last_acc)
    for_train = np.arange(1,gen,1)
    train_size = (1/2**for_train)*2**gen
    return train_size, last_acc


# In[3]:


def lr(input_bit, x1, y1):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    X = x1
    X = min_max_scaler.fit_transform(X)
    Y = y1
    list_result=[]
    crpnum = []
    for for_train in range(1,gen - 4):
        train_size = 1/(2**for_train)
        test_size = 1 - train_size
        X_train,X_test, Y_train, Y_test = train_test_split(X, Y, train_size = train_size, test_size=test_size)
        clf = LogisticRegression()
        clf.fit(X_train,Y_train)
        score = clf.score(X_test,Y_test)
        pre_Y = clf.predict(X_test)
        #####################################################################################################
        Y_test = Y_test.reshape(-1)
        Res_Xor = np.bitwise_xor(pre_Y.astype(int), Y_test.astype(int))
        result = 1 - np.average(Res_Xor)
        list_result.append(result)
        crpnum.append((1/2**for_train)*2**gen)
        #####################################################################################################
    list_result.reverse()
    crpnum.reverse()
    return crpnum, list_result


# In[4]:


def load_data(Cha, Response, test_size):
    x = Cha
    y = Response
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_size)
    return x_train, x_test, y_train, y_test


def svm_c(x_train, x_test, y_train, y_test):
    svc = SVC(kernel='rbf', class_weight='balanced',)
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)

    clf = grid.fit(x_train, y_train)
    score = grid.score(x_test, y_test)
    return score

def svm(input_bit, x1, y1):
    last_acc = []
    crpnum = []
    for for_train in range(5,14):
        train_size = 1/(2**for_train)
        test_size = 1 - train_size
        if __name__ == '__main__':
            score = svm_c(*load_data(x1, y1, test_size))
        last_acc.append(score)
        crpnum.append((1/2**for_train)*2**gen)
    last_acc.reverse()
    crpnum.reverse()
    return crpnum, last_acc


# In[5]:


#FNN
def fnn(input_bit, x1, y1):
    Cha1 = x1
    Response = y1

    x = tf.placeholder(tf.float32,[None,input_bit])
    y = tf.placeholder(tf.float32,[None,1])


    weights = {
        'layer1':tf.Variable(tf.random_normal([input_bit,300])),
        'layer2':tf.Variable(tf.random_normal([300,200])),
        'layer3':tf.Variable(tf.random_normal([200,100])),
        'layer4':tf.Variable(tf.random_normal([100,1])),
    }

    l1 = tf.matmul(x,weights['layer1'])
    l1_ = tf.nn.sigmoid(l1)
    l2 = tf.matmul(l1_,weights['layer2'])
    l2_ = tf.nn.sigmoid(l2)
    l3 = tf.matmul(l2_,weights['layer3'])
    l3_ = tf.nn.sigmoid(l3)
    l4 = tf.matmul(l3_,weights['layer4'])
    result = tf.nn.sigmoid(l4)

    prediction = tf.sign(2*result-1)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-result), reduction_indices = [1]))

    gloabl_steps = tf.Variable(0, trainable=False)
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
    correct_pred = tf.equal(prediction,2*y-1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    init = tf.global_variables_initializer()
    init_op = tf.global_variables_initializer()
    
    step_num = 1000
    batch_num = 2

    acc_best = 0.0
    last_acc = []
    train_acc = []
    crpnum = []
    with tf.Session() as sess:
        for for_train in range(1,gen-4):
            best_train_acc = 0
            best_test_acc = 0
            train_size = 1/(2**for_train)
            test_size = 1 - train_size
            x_train, x_test, y_train, y_test = ms.train_test_split(Cha1, Response, train_size=train_size, test_size=test_size)

            batch_size = x_train.shape[0]//batch_num
            sess.run(init)
            sess.run(init_op)
            for step in range(step_num):
                for batch in range(batch_num):
                    batch_x,batch_y = x_train[batch_size*batch:batch_size*(batch+1),:],y_train[batch_size*batch:batch_size*(batch+1),:]
                    sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
                acc = sess.run(accuracy,feed_dict={x:x_test,y:y_test})
                acc_train = sess.run(accuracy,feed_dict={x:x_train,y:y_train})
                loss_ = sess.run(loss,feed_dict={x:x_train,y:y_train})
                best_train_acc = acc_train
                best_test_acc = acc
                if step%100 ==0:
                    print("Step " + str(step) +" loss:"+str(loss_)+ " train Accuracy£º"  + str(acc_train) + " test accurary:"+str(acc))
                    #break
            last_acc.append(best_test_acc)
            train_acc.append(best_train_acc)
            crpnum.append((1/2**for_train)*2**gen)
    last_acc.reverse()
    crpnum.reverse()
    return crpnum, last_acc


# In[6]:


def adab(input, x1, y1):
    X = x1
    y = y1
    acc_list=[]
    crpnum = []
    for i in range(1,12):
        X_train,X_test,y_train,y_test = ms.train_test_split(X,y,train_size = 1/(2**i),test_size=1-1/(2**i))
        AdaBoost1 = AdaBoostClassifier()
        AdaBoost1.fit(X_train,y_train)
        pred1 = AdaBoost1.predict(X_test)

        acc_list.append(metrics.accuracy_score(y_test, pred1))
        crpnum.append(1/(2**i)*2**gen)
    acc_list.reverse()
    crpnum.reverse()
    return crpnum, acc_list


# In[7]:


input_bit_up   = 16
input_bit_down = 4
gen = max(input_bit_up,input_bit_down)

def o2bin(l,value):
    str1=bin(value)
    num=str1.split('b')[1]
    for i in range(l-len(num)):
        num = "0" + num
    return num

stage_up = np.random.normal(loc=0.0, scale=0.01, size=(input_bit_up,1))
stage_down = np.random.normal(loc=0.0, scale=0.01, size=(input_bit_down+1,1))
int_cha = random.sample(range(0,2**max(input_bit_up,input_bit_down)),2**gen)

def gen_fv (input_bit, challenge):
    c_temp=np.int8(list(o2bin(input_bit, challenge)))
    #print(c_temp)
    c_temp[c_temp==0]=-1
    #print(c_temp)
    return list(c_temp)

lfv  = []
lres = []
for i in int_cha:
    fv=gen_fv(max(input_bit_up,input_bit_down),i)
    #print('fv:',fv)
    lfv.append(fv)

    delay_up = np.sum([a*b for a,b in zip(fv[0:input_bit_up],stage_up[:,0])])
    res_up   = np.sign(delay_up)
    #print('res_up:',res_up)
    fv_down  = copy.deepcopy(fv)
    fv_down.insert(input_bit_down//2,res_up)
    #print("fv_down",fv_down)
    delay_down = np.sum([a*b for a,b in zip(fv_down[0:input_bit_down+1],stage_down[:,0])])
    res_down   = np.sign(delay_down)
    lres.append(res_down)
FV=np.array(lfv)
RES=0.5-0.5*np.array(lres).reshape(2**gen,1)

np.savez("iPUF-16+4.npz", FV=FV, RES=RES)


# In[ ]:


D = np.load("iPUF-16+4.npz")
FV = D['FV']
RES = D['RES']


# In[8]:


gen=16
runtimes = 10
for i in range(runtimes):
    a, b = ANN(input_bit_up, input_bit_down, FV, RES)
    totaldf = record(totaldf, 'xorapuf', 'ann', input_bit_up, i, a, b)
for i in range(runtimes):
    a, b = lr(max(input_bit_up,input_bit_down), FV, RES)
    totaldf = record(totaldf, 'xorapuf', 'lr', input_bit_up, i, a, b)
for i in range(runtimes):
    a, b = svm(max(input_bit_up,input_bit_down), FV, RES)
    totaldf = record(totaldf, 'xorapuf', 'svm', input_bit_up, i, a, b)
for i in range(runtimes):
    a, b = fnn(max(input_bit_up,input_bit_down), FV, RES)
    totaldf = record(totaldf, 'xorapuf', 'fnn', input_bit_up, i, a, b)
for i in range(runtimes):
    a, b = adab(max(input_bit_up,input_bit_down), FV, RES)
    totaldf = record(totaldf, 'xorapuf', 'adab', input_bit_up, i, a, b)
totaldf.to_csv('iPUF-16+4.csv')


