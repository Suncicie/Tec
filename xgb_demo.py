
# coding: utf-8

# In[19]:

import os
os.chdir("/Users/suncicie/Study/Project/DataMining/Competition/Tecent")# 将当前目录更换到此目录下
print os.getcwd()


# In[20]:

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import time
start_time=time.time()


# In[21]:

train=pd.read_csv("dataset/pre/train.csv")
test=pd.read_csv("dataset/pre/test.csv")


# In[26]:

del train["conversionTime"]
print train.head()


# In[27]:

# split the train data
train_xy,val=train_test_split(train,test_size=0.3,random_state=1)
y=train_xy.label
X=train_xy.drop(["label"],axis=1)
val_y=val.label
val_X=val.drop(["label"],axis=1)

# create xgb matrix
xgb_val=xgb.DMatrix(val_X,label=val_y)
xgb_train=xgb.DMatrix(X,label=y)
xgb_test=xgb.DMatrix(test)


# In[30]:

print len(train_xy)
print len(val)


# In[28]:

params={
    "booster":"gbtree",
    "objective":"multi:softprob",
    "num_class":10,
    "gamma":0.1,
    "max_depth":10,
    "lambda":2,
    "subsample":0.7,
    "colsample_bytree":0.7,
    "min_child_weight":3,
    "silent":0,
    "eta":0.007,
    "seed":1000,
    "nthread":7,
}
plst=list(params.items())
num_rounds=5000
watchlist=[(xgb_train,"train"),(xgb_val,"val")]

model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)
model.save_model('dataset/pre/xgb.model')
print "best best_ntree_limit",model.best_ntree_limit


# In[17]:

test=pd.read_csv("dataset/pre/test.csv")
sub_test=test[[test.columns[0],test.columns[1]]]
sub_test["prob"]=0
print sub_test.head()


# In[18]:

result=sub_test[["instanceID","prob"]]
print result.head()
result.to_csv("dataset/pre/result01.csv")


# In[ ]:



