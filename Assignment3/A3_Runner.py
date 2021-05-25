#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


# In[5]:


def give_results(RulesFile,TestFile):
    file=open(RulesFile,'rb')
    Rules=pickle.load(file)
    file.close()
    b= pd.read_csv(TestFile)
    df_test=pd.DataFrame(b)
    mapping = {0:"0h",1:"1h",2:"2h",3:"3h",4:"4h",5:"5h",6:"6h"}
    mapping2 = {0:"0v",1:"1v",2:"2v",3:"3v",4:"4v",5:"5v"}
    mapping3 = {1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7"}
    df_test.replace({'Horizontal_Distance_To_Hydrology': mapping, 'Vertical_Distance_To_Hydrology': mapping2},inplace=True)
    pred=["" for i in range(len(df_test))]
    df_test=pd.DataFrame(data=df_test,index=range(len(df_test)))
    for i in tqdm(range(len(df_test))):

        label="2"

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
                label=currentLabel
                break
        pred[i]=str(label)
    result=pd.DataFrame(df_test["id"])
    result["Label"]=pd.DataFrame(pred)
    result.to_csv("results.csv",index=False)


# In[9]:


give_results("A3_Rules","test_X.csv")


# In[ ]:




