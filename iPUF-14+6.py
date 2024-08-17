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

    #设置权重，先设置为1层，需要再增加
    weights = {
        'delay_up':tf.Variable(tf.random_normal([input_bit_up,1])),#tf.zeros   tf.random_normal
        'delay_down':tf.Variable(tf.random_normal([input_bit_down+1,1]))
    }

    #定义前向传播函数,得到两组RRAM的电导差
    def neural_network(x):
        delay1=tf.matmul(x[:,0:input_bit_up],weights['delay_up'])
        res_up=tf.sigmoid(delay1)
        delay2=tf.matmul(x[:,0:input_bit_down],weights['delay_down'][0:input_bit_down])+tf.multiply(res_up,weights['delay_down'][input_bit_down,0])
        res_down=tf.sigmoid(delay2)
        return res_down

    #result是电导差，result是根据电导差得到的输出结果
    #net_out差很小，是否扩大一定的倍数？使得result更接近0，1？
    result = neural_network(x)
    prediction = tf.sign(2*result-1)
    #loss函数，是方差
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - result), reduction_indices = [1]))
    #学习率 经常要调
    train_step = tf.train.AdamOptimizer(0.005).minimize(loss)

    #正确率
    correct_pred = tf.equal(prediction, 2 * y - 1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #设置初始化
    init = tf.global_variables_initializer()
    step_num = 2000
    batch_num = 1
    acc_best = 0.0
    last_acc = []

    #训练
    with tf.Session() as sess:
        for for_train in range(1,gen):#gen是在这里用的
            train_size = 1/(2**for_train)
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
            #last_train_acc.append(acc_train)
    print(last_acc)
    for_train = np.arange(1,gen,1)
    train_size = (1/2**for_train)*2**gen
    return train_size, last_acc


# In[3]:


def lr(input_bit, x1, y1):
    '''
    ÌØÕ÷¹éÒ»»¯:
        ¸÷ÌØÕ÷Ö®¼äµÄ´óÐ¡·¶Î§Ò»ÖÂ£¬²ÅÄÜÊ¹ÓÃ¾àÀë¶ÈÁ¿µÈËã·¨
        ¼ÓËÙÌÝ¶ÈÏÂ½µËã·¨µÄÊÕÁ²
        ÔÚSVMËã·¨ÖÐ£¬Ò»ÖÂ»¯µÄÌØÕ÷ÄÜ¼ÓËÙÑ°ÕÒÖ§³ÖÏòÁ¿µÄÊ±¼ä
        ²»Í¬µÄ»úÆ÷Ñ§Ï°Ëã·¨£¬ÄÜ½ÓÊÜµÄÊäÈëÊýÖµ·¶Î§²»Ò»Ñù
    '''
    '''
    sklearnÖÐ×î³£ÓÃµÄÌØÕ÷¹éÒ»»¯·½·¨MinMaxScalerºÍStandardScaler¡£
    ËüÄ¬ÈÏ½«Ã¿ÖÖÌØÕ÷µÄÖµ¶¼¹éÒ»»¯µ½[0£¬1]Ö®¼ä£¬¹éÒ»»¯ºóµÄÊýÖµ´óÐ¡·¶Î§ÊÇ¿Éµ÷µÄ£¨¸ù¾ÝMinMaxScalerµÄ²ÎÊýfeature_rangeµ÷Õû£©¡£
    ÏÂÃæ´úÂë½«ÌØÕ÷¹éÒ»»¯µ½[-1,1]Ö®¼ä¡£
    '''
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    X = x1
    #fit_transform(partData)¶Ô²¿·ÖÊý¾ÝÏÈÄâºÏfit£¬ÕÒµ½¸ÃpartµÄÕûÌåÖ¸±ê£¬Èç¾ùÖµ¡¢·½²î¡¢×î´óÖµ×îÐ¡ÖµµÈµÈ£¨¸ù¾Ý¾ßÌå×ª»»µÄÄ¿µÄ£©£¬È»ºó¶Ô¸ÃpartData½øÐÐ×ª»»transform£¬´Ó¶øÊµÏÖÊý¾ÝµÄ±ê×¼»¯¡¢¹éÒ»»¯µÈµÈ¡£¡£
    X = min_max_scaler.fit_transform(X)
    Y = y1
    list_result=[]
    crpnum = []
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
        #####################################################################################################
        #ÏÂÃæµÄ±ä»»Ö÷ÒªÊÇÎªÁËÑµÁ·Ê±£¬»®·ÖÊý¾Ý¼¯ºÍ²âÊÔ¼¯×ö×¼±¸
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
    score = grid.score(x_test, y_test)
    #print('¾«¶ÈÎª%s' % score)
    return score

def svm(input_bit, x1, y1):
    last_acc = []
    crpnum = []
    for for_train in range(5,11):#×îÐ¡µÄtrain_sizeÊÇ2**10µÄ1/2**7
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
        print('Ä£ÐÍµÄ×¼È·ÂÊÎª£º\n',metrics.accuracy_score(y_test, pred1))
        acc_list.append(metrics.accuracy_score(y_test, pred1))
        crpnum.append(1/(2**i)*2**gen)
    acc_list.reverse()
    crpnum.reverse()
    return crpnum, acc_list


# In[7]:


input_bit_up   = 14
input_bit_down = 6
gen = 14

def o2bin(l,value):
    str1=bin(value)
    num=str1.split('b')[1]
    for i in range(l-len(num)):
        num = "0" + num
    return num

stage_up = np.random.normal(loc=0.0, scale=0.01, size=(input_bit_up,1))
stage_down = np.random.normal(loc=0.0, scale=0.01, size=(input_bit_down+1,1))
int_cha = random.sample(range(0,2**max(input_bit_up,input_bit_down)),2**gen)#sample其实是采样，为了避免重复，在列表中进行采样

def gen_fv (input_bit, challenge):
    c_temp=np.int8(list(o2bin(input_bit, challenge)))
    #print(c_temp)
    c_temp[c_temp==0]=-1
    #print(c_temp)
    return list(c_temp)

lfv  = []
lres = []
#判断MUXX控制信号
for i in int_cha:
    #生成特征向量
    fv=gen_fv(max(input_bit_up,input_bit_down),i)
    #print('fv:',fv)
    lfv.append(fv)
    #生成异或前的延时值
    delay_up = np.sum([a*b for a,b in zip(fv[0:input_bit_up],stage_up[:,0])])#from 飞
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
#print(lfv)
print(RES)
print("The 1 in res is "+ str(100*RES.mean())+" %.")
np.savez("\\npzdata\\iPUF-14+6-20230320.npz", FV=FV, RES=RES)


# In[8]:


gen=14
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
totaldf.to_csv('\\iPUF-14+6-20230320.csv')


# In[9]:


crp8192=totaldf[totaldf['crp']==32767]
crp8192.tail(100)

