#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from weka.associations import Associator
from weka.core.converters import Loader
from sklearn.metrics import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import resample


# In[2]:


import weka.core.jvm as jvm
jvm.start()


# In[3]:


def func2(opt):
    a= pd.read_csv("training.csv")
    df_train=pd.DataFrame(a)
    dic={0: 'Elevation', 1: 'Aspect', 2: 'Slope', 3: 'Wilderness', 4: 'Soil_Type', 5: 'Hillshade_9am', 6: 'Hillshade_noon', 7: 'Horizontal_Distance_To_Hydrology', 8: 'Vertical_Distance_To_Hydrology', 9: 'Horizontal_Distance_To_Fire_Points', 10: 'Label'}
    
    score=[]
    mapping = {0:"0h",1:"1h",2:"2h",3:"3h",4:"4h",5:"5h",6:"6h"}
    mapping2 = {0:"0v",1:"1v",2:"2v",3:"3v",4:"4v",5:"5v"}
    mapping3 = {1:"Class 1",2:"Class 2",3:"Class 3",4:"Class 4",5:"Class 5",6:"Class 6",7:"Class 7"}
    df_train.replace({'Horizontal_Distance_To_Hydrology': mapping, 'Vertical_Distance_To_Hydrology': mapping2, 'Label':mapping3},inplace=True)
    df_train.drop(columns=['id'],inplace=True)

    ans=[]
    kf = KFold(n_splits=3,shuffle=True,random_state=2)
    features=['Elevation', 'Aspect', 'Slope', 'Wilderness', 'Soil_Type',
       'Hillshade_9am', 'Hillshade_noon', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Fire_Points',
       'Label']
    X=np.array(df_train)
    y=np.array(df_train["Label"])
    for train_index, test_index in kf.split(df_train):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train=pd.DataFrame(X_train)
        X_train.rename(columns=dic,inplace=True)
        X_test=pd.DataFrame(X_test)
        X_test.rename(columns=dic,inplace=True)
        df_1 = X_train[X_train.Label=="Class 1"]
        df_2 = X_train[X_train.Label=="Class 2"]
        df_3 = X_train[X_train.Label=="Class 3"]
        df_4 = X_train[X_train.Label=="Class 4"]
        df_5 = X_train[X_train.Label=="Class 5"]
        df_6 = X_train[X_train.Label=="Class 6"]
        df_7 = X_train[X_train.Label=="Class 7"]

        max_val=min(len(df_1),len(df_2),len(df_3),len(df_4),len(df_5),len(df_6),len(df_7))

        if(len(df_1)!=max_val):
            df_1 = resample(df_1,replace=False,n_samples=max_val,random_state=42) 
        if(len(df_2)!=max_val):
            df_2 = resample(df_2,replace=False,n_samples=max_val,random_state=42) 
        if(len(df_3)!=max_val):
            df_3 = resample(df_3,replace=False,n_samples=max_val,random_state=42) 
        if(len(df_4)!=max_val):
            df_4 = resample(df_4,replace=False,n_samples=max_val,random_state=42) 
        if(len(df_5)!=max_val):
            df_5 = resample(df_5,replace=False,n_samples=max_val,random_state=42) 
        if(len(df_6)!=max_val):
            df_6 = resample(df_6,replace=False,n_samples=max_val,random_state=42) 
        if(len(df_7)!=max_val):
            df_7 = resample(df_7,replace=False,n_samples=max_val,random_state=42) 

        X_train = pd.concat([df_1,df_2,df_3,df_4,df_5,df_6,df_7])

        X_train=X_train.reset_index()
        X_train.drop(columns=['index'],inplace=True)
        
        X_train.to_csv("train_test_split.csv",index=False)
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file("train_test_split.csv")
        associator = Associator(classname="weka.associations.Apriori",options=opt)
        associator.build_associations(data)
        a=str(associator)
        ind=a.find("Best rules found:")
        
        Rules=a[ind+len("Best rules found:\n\n  "):].split("\n")
        for i in range(len(Rules)-1):
            temp=Rules[i].split(" ")
            for start in range(len(temp)):
                if(temp[start][:len(temp[start])-1].isdigit()):
                    break
            for j in range(len(temp)):
                if(temp[j].isdigit()):
                    break

            Rules[i]=temp[start+1:j]+list(temp[j+3])
        Rules=Rules[:-1]
        covered=np.zeros(len(X_train))
        RuleFinal=[]
        
        for i in tqdm(range(len(Rules))):
            current_sup=0
            current_conf_num=0
            currentRule=Rules[i][0:-1]
            currentLabel=Rules[i][-1]
            if(len(Rules[i])>3):
                for j in range(len(X_train)):
                    if(covered[j]==0):
                        TempLabel=True
                        for k in currentRule:
                            temp=k.split("=")
                            if(X_train[temp[0]][j]!=temp[1]):
                                TempLabel=False                
                                break
                        if(TempLabel):
                            current_sup+=1
                            if("Class "+currentLabel==X_train["Label"][j]):
                                current_conf_num+=1
                                covered[j]=1
            if(current_conf_num>0):
                RuleFinal.append(Rules[i])
        Rules=RuleFinal
        pred=["" for i in range(len(y_test))]
        df_test=pd.DataFrame(data=X_test,index=range(len(X_test)))
        for i in range(len(df_test)):
            label="Class 1"
            for j in range(len(Rules)):

                currentRule=Rules[j][0:-1]
                currentLabel=Rules[j][-1]
                TempLabel=True
                for k in currentRule:
                    temp=k.split("=")
                    if(df_test[temp[0]][i]!=temp[1]):
                        TempLabel=False                
                        break
                if(TempLabel):
                    label="Class "+currentLabel
                    break
            pred[i]=str(label)
        score.append(f1_score(y_test,pred,average="macro"))
        print(f1_score(y_test,pred,average="macro"))
    return(score)


# In[4]:


score=np.array(func2(['-N', '100', '-A', '-C', '0.45', '-M', '0.001']))


# In[6]:


score.mean()


# In[7]:


score


# In[ ]:




