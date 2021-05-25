#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import math
import numpy
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


# In[37]:


def PreProcess():
    data_train = pd.read_csv('given_dataset.csv')    
    Training_Feature=['A','B','C','D','E','F','G']
    x=data_train[Training_Feature]
    y=data_train['T']
    #print("Skewness before preprocessing")
    # print(x.skew(axis = 0))
    # x['A']=numpy.log(x['A'])
    # x['B']=numpy.log(x['B'])
    # x['E']=numpy.cbrt(x['E'])
    # x['F']=numpy.cbrt(x['F'])
    # x['G']=numpy.cbrt(x['G'])
    #print("Skewness after preprocessing")
    # print(x.skew(axis = 0))
    undersample = RandomUnderSampler(sampling_strategy="majority")
    x_under, y_under = undersample.fit_resample(x, y)
    return(x_under,y_under)


# In[39]:


def Prediction1(ToPredict,x_under,y_under):
    data_test = pd.read_csv('to_predict.csv')
    Training_Feature=['A','B','C','D','E','F','G']
    x_test=data_test[Training_Feature]
    model1=GradientBoostingClassifier(max_depth=8,n_estimators=300,max_features=4)
    model1.fit(x_under,y_under)
    pred=model1.predict(x_test)
    predict_df=pd.DataFrame(data=pred)
    predict_df['id']=data_test['id']
    predict_df.rename({0:'T'},axis=1,inplace=True)
    predict_df['T'].value_counts()
    predict_df.to_csv('Model1Prediction.csv',index=False,columns=['id','T'])
    pickle.dump(model1, open("Model1Prediction", 'wb'))


# In[40]:


def Prediction2(ToPredict,x_under,y_under):
    data_test = pd.read_csv('to_predict.csv')
    Training_Feature=['A','B','C','D','E','F','G']
    x_test=data_test[Training_Feature]
    model1=GradientBoostingClassifier(max_depth=8,n_estimators=300,max_features=4)
    model1.fit(x_under,y_under)
    pred=model1.predict_proba(x_test)[:,1]
    predict_df=pd.DataFrame(data=pred)
    predict_df['id']=data_test['id']
    predict_df.rename({0:'T'},axis=1,inplace=True)
    predict_df['T'].value_counts()
    predict_df.to_csv('Model2Prediction.csv',index=False,columns=['id','T'])
    pickle.dump(model1, open("Model2Prediction", 'wb'))


# In[41]:


def Prediction3(ToPredict,x_under,y_under):
    data_test = pd.read_csv('to_predict.csv')
    Training_Feature=['A','B','C','D','E','F','G']
    x_test=data_test[Training_Feature]
    model1=RandomForestClassifier(bootstrap=True,max_samples=40000,max_depth= 20, max_features= 4, n_estimators= 100, warm_start= True)
    model1.fit(x_under,y_under)
    pred=model1.predict(x_test)
    predict_df=pd.DataFrame(data=pred)
    predict_df['id']=data_test['id']
    predict_df.rename({0:'T'},axis=1,inplace=True)
    predict_df['T'].value_counts()
    predict_df.to_csv('Model3Prediction.csv',index=False,columns=['id','T'])
    pickle.dump(model1, open("Model3Prediction", 'wb'))


# In[42]:


def LoadModel(ToPredict,ModelName):
    loaded_model = pickle.load(open(ModelName, 'rb'))
    data_test = pd.read_csv('to_predict.csv')
    Training_Feature=['A','B','C','D','E','F','G']
    x_test=data_test[Training_Feature]
    pred=loaded_model.predict(x_test)
    predict_df=pd.DataFrame(data=pred)
    predict_df['id']=data_test['id']
    predict_df.rename({0:'T'},axis=1,inplace=True)
    predict_df['T'].value_counts()
    predict_df.to_csv('LoadedModelPrediction.csv',index=False,columns=['id','T'])
    


# In[43]:


def LoadModelProbab(ToPredict,ModelName):
    loaded_model = pickle.load(open(ModelName, 'rb'))
    data_test = pd.read_csv('to_predict.csv')
    Training_Feature=['A','B','C','D','E','F','G']
    x_test=data_test[Training_Feature]
    pred=loaded_model.predict_proba(x_test)[:,1]
    predict_df=pd.DataFrame(data=pred)
    predict_df['id']=data_test['id']
    predict_df.rename({0:'T'},axis=1,inplace=True)
    predict_df['T'].value_counts()
    predict_df.to_csv('LoadedModelPrediction.csv',index=False,columns=['id','T'])
    


# In[22]:


# x_under,y_under=PreProcess()


# # In[48]:


# Prediction1("to_predict.csv",x_under,y_under)


# # In[35]:


# LoadModel("to_predict.csv","Model1Prediction")
