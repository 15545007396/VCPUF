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
import random
import pandas as pd
import sklearn
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
import sklearn.metrics as metrics

totaldf = pd.DataFrame(columns=['puf','algo','size','run','crp','train','test'])
def record(totaldf, puf, algo, size, run, crp, train, test):
    tempdf = pd.DataFrame({
    'puf':puf,
    'algo':algo,
    'size':size,
    'run': run,
    'crp': crp,
    'train': train,
    'test': test})
    new_totaldf = pd.concat([totaldf, tempdf],ignore_index=True)
    return new_totaldf


# In[2]:


def ANN(input_bit, FV, RES):
    rram_num = np.int32(2**(input_bit/2))
    total_crp = rram_num*(rram_num-1)/2
    #占位，同时规定了输入输出的行列数
    x = tf.placeholder(tf.float32,[None,rram_num])
    y = tf.placeholder(tf.float32,[None,1])

    weights = {'hidden_1':tf.Variable(tf.random_normal([rram_num,1]))}

    def neural_network(x):
        hidden_layer_1 = tf.matmul(x,weights['hidden_1'])
        #l1 = tf.nn.tanh(hidden_layer_1)
        return hidden_layer_1

    #result是电导差，result是根据电导差得到的输出结果
    #net_out差很小，是否扩大一定的倍数？使得result更接近0，1？
    result = neural_network(x)
    prediction = tf.sign(2*result-1)
    #loss函数，是方差
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - result), reduction_indices = [1]))
    #学习率 经常要调
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    #正确率
    correct_pred = tf.equal(prediction, 2 * y - 1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #设置初始化
    init = tf.global_variables_initializer()
    step_num = 1500
    batch_num = 1
    acc_best = 0.0
    last_acc = []
    last_train_acc = []
    train_crp = []
    #训练
    with tf.Session() as sess:
        for for_train in range(1,gen-2):#gen是在这里用的
            train_size = 1/(2**for_train)
            train_crp.append(total_crp*train_size)
            test_size = 1 - train_size
            x_train, x_test, y_train, y_test = ms.train_test_split(FV, RES, train_size = train_size, test_size = test_size)
            batch_size = x_train.shape[0]//batch_num
            sess.run(init)#初始化
            acc = sess.run(accuracy, feed_dict = {x:x_test,y:y_test})
            #print("train_size:" + str(train_size) + "initial acc" + str(acc))
            #print(np.mean(y_test), x_train.shape)#这里把np.mean(response)去掉了 
            for step in range(step_num + 1):
                for batch in range(batch_num):
                    batch_x, batch_y = x_train[batch_size*batch:batch_size*(batch + 1), : ], y_train[batch_size*batch:batch_size*(batch + 1), : ]
                    sess.run(train_step, feed_dict = {x:batch_x,y:batch_y})
                    #print(result)
                acc = sess.run(accuracy,feed_dict = {x:x_test, y:y_test})
                acc_train = sess.run(accuracy,feed_dict = {x:x_train, y:y_train})
                loss_ = sess.run(loss,feed_dict = {x:x_train,y:y_train})
                if step%100 == 0:
                    print("Step " + str(step) +" loss:"+str(loss_)+ " train Accuracy："  + str(acc_train) + " test accurary:"+str(acc))
            last_acc.append(acc)
            last_train_acc.append(acc_train)
    print(last_acc)
    return train_crp, last_train_acc, last_acc


# In[3]:


def lr(input_bit, x1, y1):
    rram_num = np.int32(2**(input_bit/2))
    total_crp = rram_num*(rram_num-1)/2
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    X = x1
    #fit_transform(partData)¶Ô²¿·ÖÊý¾ÝÏÈÄâºÏfit£¬ÕÒµ½¸ÃpartµÄÕûÌåÖ¸±ê£¬Èç¾ùÖµ¡¢·½²î¡¢×î´óÖµ×îÐ¡ÖµµÈµÈ£¨¸ù¾Ý¾ßÌå×ª»»µÄÄ¿µÄ£©£¬È»ºó¶Ô¸ÃpartData½øÐÐ×ª»»transform£¬´Ó¶øÊµÏÖÊý¾ÝµÄ±ê×¼»¯¡¢¹éÒ»»¯µÈµÈ¡£¡£
    X = min_max_scaler.fit_transform(X)
    Y = y1
    list_result=[]
    crpnum = []
    train_result=[]
    for for_train in range(1,gen - 4):
        train_size = 1/(2**for_train)
        test_size = 1 - train_size
        X_train,X_test, Y_train, Y_test = train_test_split(X, Y, train_size = train_size, test_size=test_size)
        #ÏÂÃæ¿ªÊ¼µ÷ÓÃsklearnµÄÑµÁ·LRº¯Êý
        clf = LogisticRegression()
        clf.fit(X_train,Y_train)
        #scoreÐ£Ñé
        score = clf.score(X_test,Y_test)
        #½øÐÐÔ¤²â
        pre_Y = clf.predict(X_test)
        pre_train_Y = clf.predict(X_train)
        #####################################################################################################
        #ÏÂÃæµÄ±ä»»Ö÷ÒªÊÇÎªÁËÑµÁ·Ê±£¬»®·ÖÊý¾Ý¼¯ºÍ²âÊÔ¼¯×ö×¼±¸
        Y_test = Y_test.reshape(-1)
        Y_train = Y_train.reshape(-1)
        train_xor = np.bitwise_xor(pre_train_Y.astype(int), Y_train.astype(int))
        Res_Xor = np.bitwise_xor(pre_Y.astype(int), Y_test.astype(int))
        result = 1 - np.average(Res_Xor)
        result_train = 1 - np.average(train_xor)
        train_result.append(result_train)
        list_result.append(result)
        crpnum.append(total_crp*train_size)
        #####################################################################################################
    list_result.reverse()
    crpnum.reverse()
    train_result.reverse()
    return crpnum, train_result, list_result


# In[4]:


def load_data(Cha, Response, test_size):
    #x = data[:, 1:]  # Êý¾ÝÌØÕ÷
    #y = data[:, 0].astype(int)  # ±êÇ©
    x = Cha
    y = Response
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)  # ±ê×¼»¯
    # ½«Êý¾Ý»®·ÖÎªÑµÁ·¼¯ºÍ²âÊÔ¼¯£¬test_size=.3±íÊ¾30%µÄ²âÊÔ¼¯
    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_size)
    #print(x_train)
    #print(y_train)
    return x_train, x_test, y_train, y_test


def svm_c(x_train, x_test, y_train, y_test):
    # rbfºËº¯Êý£¬ÉèÖÃÊý¾ÝÈ¨ÖØ
    svc = SVC(kernel='rbf', class_weight='balanced',)
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # Íø¸ñËÑË÷½»²æÑéÖ¤µÄ²ÎÊý·¶Î§£¬cv=3,3ÕÛ½»²æ
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # ÑµÁ·Ä£ÐÍ
    clf = grid.fit(x_train, y_train)
    # ¼ÆËã²âÊÔ¼¯¾«¶È
    score_train = grid.score(x_train, y_train)
    score = grid.score(x_test, y_test)
    #print('¾«¶ÈÎª%s' % score)
    return score_train, score

def svm(input_bit, x1, y1):
    rram_num = np.int32(2**(input_bit/2))
    total_crp = rram_num*(rram_num-1)/2
    last_acc = []
    train_acc = []
    crpnum = []
    for for_train in range(3,8):#×îÐ¡µÄtrain_sizeÊÇ2**10µÄ1/2**7
        train_size = 1/(2**for_train)
        test_size = 1 - train_size
        if __name__ == '__main__':
            score_train, score_test = svm_c(*load_data(x1, y1, test_size))
        last_acc.append(score_test)
        train_acc.append(score_train)
        crpnum.append(total_crp*train_size)
    last_acc.reverse()
    crpnum.reverse()
    train_acc.reverse()
    return crpnum, train_acc, last_acc


# In[5]:


#FNN
def fnn(input_bit, x1, y1):
    rram_num = np.int32(2**(input_bit/2))
    total_crp = rram_num*(rram_num-1)/2
    
    Cha1 = x1
    Response = y1

    x = tf.placeholder(tf.float32,[None,rram_num])
    y = tf.placeholder(tf.float32,[None,1])


    weights = {
        'layer1':tf.Variable(tf.random_normal([rram_num,300])),
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
    #loss = tf.reduce_mean(tf.square(result-y))+tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(4e-9), tf.trainable_variables())
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-result), reduction_indices = [1]))

    gloabl_steps = tf.Variable(0, trainable=False)
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
    correct_pred = tf.equal(prediction,2*y-1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    init = tf.global_variables_initializer()
    init_op = tf.global_variables_initializer()
    print(weights)
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
                    #sess.run(clip_op_data)
                    #sess.run(clip_op_select)
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
            crpnum.append(total_crp*train_size)
    last_acc.reverse()
    crpnum.reverse()
    train_acc.reverse()
    return crpnum, train_acc, last_acc


# In[6]:


def adab(input, x1, y1):
    rram_num = np.int32(2**(input_bit/2))
    total_crp = rram_num*(rram_num-1)/2
    X = x1
    y = y1
    acc_list=[]
    acc_train=[]
    crpnum = []
    for i in range(1,12):
        X_train,X_test,y_train,y_test = ms.train_test_split(X,y,train_size = 1/(2**i),test_size=1-1/(2**i))
        AdaBoost1 = AdaBoostClassifier()
        AdaBoost1.fit(X_train,y_train)
        pred1 = AdaBoost1.predict(X_test)
        pred2 = AdaBoost1.predict(X_train)
        print('Ä£ÐÍµÄ×¼È·ÂÊÎª£º\n',metrics.accuracy_score(y_test, pred1))
        acc_list.append(metrics.accuracy_score(y_test, pred1))
        acc_train.append(metrics.accuracy_score(y_train, pred2))
        crpnum.append(total_crp*1/(2**i))
    acc_list.reverse()
    crpnum.reverse()
    acc_train.reverse()
    return crpnum, acc_train, acc_list


# In[9]:


input_bit = 14
rram_num = np.int32(2**(input_bit/2))

f = open("/home/cyl/eda/tensorenv/VCPUF/arrayPUF/rram_test.lis","rb")

#ÒÔÏÂÉú³ÉresponseÊý¾Ý
lines1 =  f.read().decode("utf-8")
lines = lines1.split("\n")
regs = []
for line in lines:
    strings = line.split(" ")
    try:
        if strings[2] == "1.0000e-07":
            regs.append(np.float32(strings[6]))
    except:
        pass
regs = np.array(regs)
response = []
xi1 = []
xi2 = []
rram_select = np.random.choice(regs,rram_num,replace=True)
for i in range(0,len(rram_select)):
    for j in range(i+1,len(rram_select)):
        if i!=j:
            xi1.append(i)
            xi2.append(j)
            if rram_select[i]>rram_select[j]:
                response.append(1);
            else:
                response.append(0);

response_num = len(response)
response = np.array([response])
RES = response.reshape(response_num,1)
print(len(RES))
x1 = np.zeros([response_num,rram_num],dtype=np.float32)
nums = []
index = 0
for i,j in zip(xi1,xi2):
    x1[index,i] = 1
    x1[index,j] = -1
    index = index + 1 
print("generate finished!!") 
print(RES.mean())
FV = np.float32(x1).reshape(len(RES),rram_num)
np.savez('2cell-14bit-'+str(datetime.date.today())+'.npz', FV=FV, RES=RES)


# In[11]:


D = np.load("2cell-14bit-2023-01-12.npz")
FV = D['FV']
RES = D['RES']


# In[12]:


input_bit=14
gen=14
runtimes = 10
for i in range(runtimes):
    m, a, b = ANN(input_bit, FV, RES)
    totaldf = record(totaldf, '2cell', 'ann', input_bit, i, m, a, b)
for i in range(runtimes):
    m, a, b = lr(input_bit, FV, RES)
    totaldf = record(totaldf, '2cell', 'lr', input_bit, i, m, a, b)
for i in range(runtimes):
    m, a, b = svm(input_bit, FV, RES)
    totaldf = record(totaldf, '2cell', 'svm', input_bit, i, m, a, b)
for i in range(runtimes):
    m, a, b = fnn(input_bit, FV, RES)
    totaldf = record(totaldf, '2cell', 'fnn', input_bit, i, m, a, b)
for i in range(runtimes):
    m, a, b = adab(input_bit, FV, RES)
    totaldf = record(totaldf, '2cell', 'adab', input_bit, i, m, a, b)
totaldf.to_csv('2cell-14bit-'+str(datetime.date.today())+'.csv')


# In[ ]:




