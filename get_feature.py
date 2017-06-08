
# coding: utf-8

# In[1]:

import pandas as pd
import os
os.chdir("/Users/suncicie/Study/Project/DataMining/Competition/Tecent")# 将当前目录更换到此目录下
print os.getcwd()


# In[24]:

# read data
# connect train and user information by userID
train=pd.read_csv("dataset/pre/train.csv")
user=pd.read_csv("dataset/pre/user.csv")
position=pd.read_csv("dataset/pre/position.csv")
test=pd.read_csv("dataset/pre/test.csv")
ad=pd.read_csv("dataset/pre/ad.csv")


# In[28]:

del test["label"]
del train["conversionTime"]


# In[33]:

m_train_user=pd.merge(train,user,how="left",on="userID")
m_test_user=pd.merge(test,user,how="left",on="userID")


# In[48]:

m_train_user_position=pd.merge(m_train_user,position,how="left",on="positionID")
m_test_user_position=pd.merge(m_test_user,position,how="left",on="positionID")


# In[51]:

m_train_user_position_ad=pd.merge(m_train_user_position,ad,how="left",on="creativeID")
m_test_user_position_ad=pd.merge(m_test_user_position,ad,how="left",on="creativeID")


# In[57]:

print len(m_test_user_position_ad.columns)
print m_test_user_position_ad.columns
print len(m_train_user_position_ad.columns)
print m_train_user_position_ad.columns


# In[58]:

# save the feature data frame
m_train_user_position_ad.to_csv("dataset/pre/train_user_position_ad.csv")
m_test_user_position_ad.to_csv("dataset/pre/test_user_position_ad.csv")


# In[ ]:



