#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


# In[28]:


def func(FileName,model):
    data = pd.DataFrame(pd.read_csv(FileName))
    data.drop(columns=['id'],inplace=True)
    Ordinal_Attr=["Elevation","Aspect","Slope","Hillshade_9am","Hillshade_noon","Horizontal_Distance_To_Fire_Points"]
    Nominal_Attr=['Wilderness','Soil_Type']
    Integer_Attr=["Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology"]
    ordinal_list=[['elevation_low','elevation_medium', 'elevation_high', 'elevation_ultra'],['aspect_low','aspect_medium', 'aspect_high', 'aspect_ultra'],['slope_low','slope_medium','slope_high', 'slope_ultra'],['hillshade_9am_min', 'hillshade_9am_max'],['hillnoon_min', 'hillnoon_max'],["low","mid","high"]]
    ord_enc = OrdinalEncoder(categories=ordinal_list)
    data_ordinal = pd.DataFrame(ord_enc.fit_transform(data[Ordinal_Attr]))
    one_enc = OneHotEncoder()
    data_nominal=pd.DataFrame(one_enc.fit_transform(data[Nominal_Attr]).toarray())
    data_integers=data.drop(columns=Nominal_Attr+Ordinal_Attr)
    data_final=data_nominal
    data_final[data_integers.columns]=data_integers
    data_final[Ordinal_Attr]=data_ordinal
    pca_reduce = PCA(n_components=20)
    X_pca = pd.DataFrame(pca_reduce.fit_transform(data_final))
    pca_reduce = PCA(n_components=2)
    X_pca2 = pd.DataFrame(pca_reduce.fit_transform(data_final))
    if(model==1):
        tech= KMeans( init="random",n_clusters=7, max_iter=1000,  n_init=10,random_state=0 )
        y=pd.DataFrame(tech.fit_predict(X_pca))
    
    elif(model==2):
        tech= KMeans( init="random",n_clusters=7, max_iter=100,  n_init=100,random_state=0 )
        y=pd.DataFrame(tech.fit_predict(X_pca))
    
    elif(model==3):
        tech = KMeans( init="random",n_clusters=5, max_iter=1000,  n_init=10,random_state=0 )
        y=pd.DataFrame(tech.fit_predict(X_pca))
        
    elif(model==4):
        tech = KMeans( init="random",n_clusters=9, max_iter=1000,  n_init=10,random_state=0 )
        y=pd.DataFrame(tech.fit_predict(X_pca))
        
    elif(model==5):
        tech = KMeans( init="k-means++",n_clusters=5, max_iter=1000,  n_init=10,random_state=0 )
        y=pd.DataFrame(tech.fit_predict(X_pca))
        
    elif(model==6):
        tech = KMeans( init="k-means++",n_clusters=7, max_iter=1000,  n_init=10,random_state=0 )
        y=pd.DataFrame(tech.fit_predict(X_pca))

    elif(model==7):
        tech = KMeans( init="k-means++",n_clusters=9, max_iter=1000,  n_init=10,random_state=0 )
        y=pd.DataFrame(tech.fit_predict(X_pca))
        
    elif(model==8):
        tech = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage="average")
        y=pd.DataFrame(tech.fit_predict(X_pca2))
    
    elif(model==9):
        tech = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
        y=pd.DataFrame(tech.fit_predict(X_pca))

    elif(model==10):
        tech = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='complete')
        y=pd.DataFrame(tech.fit_predict(X_pca))

    # temp=list(y.value_counts())
    # temp.sort()
    # print(temp)
    y.to_csv("result.csv",index=False)


# In[34]:


func("test_X.csv",1)


# In[ ]:




