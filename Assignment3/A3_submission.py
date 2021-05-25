#!/usr/bin/env python
# coding: utf-8

# # Importing the libaries

# In[37]:


import numpy as np
import pandas as pd
from weka.associations import Associator
from weka.core.converters import Loader
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import resample


# In[24]:


import weka.core.jvm as jvm
jvm.start()


# # Reading the drain data

# In[25]:


a= pd.read_csv("training.csv")
df_train=pd.DataFrame(a)


# # Converting the numeric attributes to string type

# In[26]:


mapping = {0:"0h",1:"1h",2:"2h",3:"3h",4:"4h",5:"5h",6:"6h"}
mapping2 = {0:"0v",1:"1v",2:"2v",3:"3v",4:"4v",5:"5v"}
mapping3 = {1:"Class 1",2:"Class 2",3:"Class 3",4:"Class 4",5:"Class 5",6:"Class 6",7:"Class 7"}

df_train.replace({'Horizontal_Distance_To_Hydrology': mapping, 'Vertical_Distance_To_Hydrology': mapping2, 'Label':mapping3},inplace=True)
df_train.drop(columns=['id'],inplace=True)


# # Sampling the data (Balancing)

# In[27]:


df_1 = df_train[df_train.Label=="Class 1"]
df_2 = df_train[df_train.Label=="Class 2"]
df_3 = df_train[df_train.Label=="Class 3"]
df_4 = df_train[df_train.Label=="Class 4"]
df_5 = df_train[df_train.Label=="Class 5"]
df_6 = df_train[df_train.Label=="Class 6"]
df_7 = df_train[df_train.Label=="Class 7"]

max_val=min(len(df_1),len(df_2),len(df_3),len(df_4),len(df_5),len(df_6),len(df_7))

if(len(df_1)!=max_val):
    df_1 = resample(df_1,replace=False,n_samples=max_val,random_state=41) 
if(len(df_2)!=max_val):
    df_2 = resample(df_2,replace=False,n_samples=max_val,random_state=41) 
if(len(df_3)!=max_val):
    df_3 = resample(df_3,replace=False,n_samples=max_val,random_state=41) 
if(len(df_4)!=max_val):
    df_4 = resample(df_4,replace=False,n_samples=max_val,random_state=41) 
if(len(df_5)!=max_val):
    df_5 = resample(df_5,replace=False,n_samples=max_val,random_state=41) 
if(len(df_6)!=max_val):
    df_6 = resample(df_6,replace=False,n_samples=max_val,random_state=41) 
if(len(df_7)!=max_val):
    df_7 = resample(df_7,replace=False,n_samples=max_val,random_state=41) 

print(df_train.Label.value_counts())
df_train = pd.concat([df_1,df_2,df_3,df_4,df_5,df_6,df_7])
print(df_train.Label.value_counts())
df_train=df_train.reset_index()
df_train.drop(columns=['index'],inplace=True)


# ### Storing the train data into a temporary file so that rules can be extracted using weka- wrapper
# ### Apriori Algorithm used

# In[28]:


df_train.to_csv("a3_submission.csv",index=False)
loader = Loader(classname="weka.core.converters.CSVLoader")
data = loader.load_file("a3_submission.csv")
associator = Associator(classname="weka.associations.Apriori",options=['-N', '1000', '-A', '-C', '0.45', '-M', '0.001'])
associator.build_associations(data)
# print(associator)


# In[29]:


asso=str(associator)
ind=asso.find("Best rules found:")
Rules=asso[ind+len("Best rules found:\n\n  "):].split("\n")
for i in tqdm(range(len(Rules)-1)):
    temp=Rules[i].split(" ")
    for start in range(len(temp)):
        if(temp[start][:len(temp[start])-1].isdigit()):
            break
    for j in range(len(temp)):
        if(temp[j].isdigit()):
            break            
    Rules[i]=temp[start+1:j]+list(temp[j+3])
Rules=Rules[:-1]
print(len(Rules))


# # Pruning the rules according to algorithm mentioned in the report

# In[30]:


covered=np.zeros(len(df_train))
RuleFinal=[]
for i in tqdm(range(len(Rules))):
    if(len(Rules[i])>3):
        current_sup=0
        current_conf_num=0
        currentRule=Rules[i][0:-1]
        currentLabel=Rules[i][-1]
        for j in range(len(df_train)):
            if(covered[j]==0):
                TempLabel=True
                for k in currentRule:
                    temp=k.split("=")
                    if(df_train[temp[0]][j]!=temp[1]):
                        TempLabel=False                
                        break
                if(TempLabel):
                    current_sup+=1
                    if("Class "+currentLabel==df_train["Label"][j]):
                        current_conf_num+=1
                        covered[j]=1
        if(current_conf_num>0):
            RuleFinal.append(Rules[i])

print(len(Rules),len(RuleFinal))
Rules=RuleFinal


# # Dumping the Rules

# In[31]:


import pickle
file=open("A3_Rules",'wb')
pickle.dump(Rules,file)
file.close()


# # Open the testing file and preproess it like the train file

# In[32]:


b= pd.read_csv("test_X.csv")
df_test=pd.DataFrame(b)
mapping = {0:"0h",1:"1h",2:"2h",3:"3h",4:"4h",5:"5h",6:"6h"}
mapping2 = {0:"0v",1:"1v",2:"2v",3:"3v",4:"4v",5:"5v"}
mapping3 = {1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7"}
df_test.replace({'Horizontal_Distance_To_Hydrology': mapping, 'Vertical_Distance_To_Hydrology': mapping2},inplace=True)
df_test.Horizontal_Distance_To_Hydrology.value_counts()


# In[33]:


pred=["" for i in range(len(df_test))]
df_test=pd.DataFrame(data=df_test,index=range(len(df_test)))
for i in tqdm(range(len(df_test))):
    
    label="2"
    
    for j in range(len(Rules)):

        currentRule=Rules[j][0:-1]
        currentLabel=Rules[j][-1]
#         print(currentRule)
#         print(currentLabel)
        TempLabel=True
        for k in currentRule:
            temp=k.split("=")
#             print(df_test[temp[0]][i+trainsize]==temp[1])
            if(df_test[temp[0]][i]!=temp[1]):
                TempLabel=False                
                break
            
        if(TempLabel):
            label=currentLabel
            break
    pred[i]=str(label)


# # Storing the results

# In[36]:


result=pd.DataFrame(df_test["id"])
result["Label"]=pd.DataFrame(pred)
result.to_csv("A3_output.csv",index=False)


# In[ ]:




