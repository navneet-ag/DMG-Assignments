# DMG 2020
# Assignment 1
# Yash Kumar Atri
# •
# 6:52 PM
# 100 points
# Due Sep 21, 11:59 PM
# Topic: Exploratory Data Analysis (EDA)

# Assn1.py
# Text

# CSE506_Assn1.pdf
# PDF
# Class comments


# Your work
# Assigned
# Private comments
# Navneet Agarwal | 2018348
# Aditya Singh |2018378

# # -*- coding: utf-8 -*-
#     """
#     Import Statements
#     """
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
from pandas.io.json import json_normalize
from datetime import datetime
from pandas.core.indexes.base import Index
# #... rest of the imports
pd.options.mode.chained_assignment = None
union_terr=['jk','an','ld','ch','dn','la','py','tt','un','dd']#union territories assuming ch->Chandigarh

State_Mappings={'ap':"Andhra Pradesh",'ar':"Arunachal Pradesh", 'as':"Assam", 'br':"Bihar", 'ct':"Chattisgarh", 'dl':"Delhi", 'ga':"Goa", 'gj':"Gujarat", 'hp':"Himachal Pradesh", 'hr':"Haryana", 'jh':"Jharkhand", 'ka':"Karnataka",'kl':"Kerala", 'mh':"Maharashtra", 'ml':"Meghalaya", 'mn':"Manipur", 'mp':"Madhya Pradesh", 'mz':"Mizoram", 'nl':"Nagaland", 'or':"Odisha", 'pb':"Punjab", 'rj':"Rajasthan", 'sk':"Sikkim", 'tg':"Telangana",'tn':"Tamil Nadu", 'tr':"Tripura", 'up':"Uttar Pradesh", 'ut':"Uttarakhand", 'wb':"WestBengal"}
def greater(start_date,end_date):
    if(start_date>end_date):
        print("Start Date should be smaller than the End Date")
        return False
    else:
        return True
def validate(date_string,format):
    try:
        datetime.strptime(date_string, format)
        return True
    except ValueError:
        print("This is the incorrect date string format. It should be YYYY-MM-DD")
        return False
def Data_frame(json_file_path, start_date, end_date):
    boo=True
    df=pd.read_json(json_file_path,orient='columns')
    df2 = pd.json_normalize(df['states_daily'])
    i=0
    for date_input in df2['date']:
        date_obj = datetime.strptime(date_input, '%d-%b-%y')
        date_final=date_obj.strftime('%Y-%m-%d')
        # print(date_final)
        df2.at[i, "date"] = date_final
        i=i+1
    if start_date not in df2.values : 
        print("\nThe Start Date does not exists in Dataframe")
        boo=False
    if end_date not in df2.values : 
        print("\nThe End Date does not exists in Dataframe")
        boo=False
    rslt_df = df2.loc[(df2['date'] >= start_date) & (df2['date'] <= end_date)]
    rslt_df = rslt_df.apply(pd.to_numeric, errors='ignore')    
    
    return (rslt_df,boo)

def get_states(df):
    df = df.drop(union_terr,axis=1)    
    df['tt'] = df.sum(axis=1)
    return df
    
def get_sorted(df,tt):
    conf = df['status'] == "Confirmed"
    reco = df['status'] == "Recovered"
    dece = df['status'] == "Deceased"
    df_conf=df[conf]
    tt_x=tt+'_x'
    tt_y=tt+'_y'#date tt tt_x tt_y
    df_conf.drop(df_conf.columns.difference(['date',tt]), 1, inplace=True)
    # print(df_conf.head())
    df_reco=df[reco]
    df_reco.drop(df_reco.columns.difference(['date',tt]), 1, inplace=True)
    # print(df_reco.head())
    df_dece=df[dece]

    df_dece.drop(df_dece.columns.difference(['date',tt]), 1, inplace=True)
    # print(df_dece.head())
    final_df=df_conf.merge(df_reco,on='date').merge(df_dece,on='date')
    # print(final_df.head())
    final_df.rename(columns = {tt_x:'Confirmed', tt_y:'Recovered', 
                              tt:'Deceased'}, inplace = True)
    return final_df

def GetValueMax(sum_column):
    final_confirm=sum_column.drop(labels=['date', 'status','tt'])
    index_conf=final_confirm.max()
    indices=[]
    if type(Index(final_confirm).get_loc(index_conf)) == int:
        # print('i am here')
        ind=Index(final_confirm).get_loc(index_conf)
    # print(ind)
        lis_conf=final_confirm.index.tolist()#multiple
        indices.append(ind)
    else:
        ind=Index(final_confirm).get_loc(index_conf).tolist()
        # print(ind)
        lis_conf=final_confirm.index.tolist()
        indices = [i for i, x in enumerate(ind) if x == True]
    return indices,lis_conf,final_confirm

def GetValueMin(sum_column):
    final_confirm=sum_column.drop(labels=['date', 'status','tt'])
    index_conf=final_confirm.min()
    indices=[]
    if type(Index(final_confirm).get_loc(index_conf)) == int:
        # print('i am here')
        ind=Index(final_confirm).get_loc(index_conf)
    # print(ind)
        lis_conf=final_confirm.index.tolist()#multiple
        indices.append(ind)
    else:
        ind=Index(final_confirm).get_loc(index_conf).tolist()
        # print(ind)
        lis_conf=final_confirm.index.tolist()
        indices = [i for i, x in enumerate(ind) if x == True]
    return indices,lis_conf,final_confirm
def LinearRegression(X,Y,Title,Date):
    X_Mean=X.mean()
    Y_Mean=Y.mean()
#     print(X_Mean,Y_Mean)
    Numerator=((X.sub(X_Mean))*(Y.sub(Y_Mean))).sum()
    Denominator=((X.sub(X_Mean))*(X.sub(X_Mean))).sum() 
#     print(Numerator,Denominator)
    Slope=(Numerator/Denominator)
    Intercept=Y_Mean - Slope*X_Mean
    print("The slope is "+str(Slope)+" and the intercept is " + str(Intercept) +" for the Linear Regression line for "+Title +" cases")    
# Plotting
    # Predicted_Y= Slope*X + Intercept
    
    # plt.scatter(Date,Y,color="green")
    # plt.plot([min(X), max(X)], [min(Predicted_Y), max(Predicted_Y)], color='blue')
    # plt.title('Plot for '+Title+" Cases")
    # plt.ylabel('Number of '+Title+' Cases')
    # plt.xlabel('Dates')
    # plt.savefig(Title+".png", dpi=300)
    # plt.show()
    return(Slope,Intercept)


def Q1_1(json_file_path, start_date, end_date):    
    if(validate(start_date,"%Y-%m-%d")==False):
        return
    if(validate(end_date,"%Y-%m-%d")==False):
        return
    if(greater(start_date,end_date)==False):
        return

    confirmed_count=0
    recovered_count=0
    deceased_count=0
    df,boo=Data_frame(json_file_path,start_date, end_date)
    if(boo==False):
        return

    final_df=get_sorted(df,'tt')
    confirmed_count=final_df['Confirmed'].sum(axis=0)
    deceased_count=final_df['Deceased'].sum(axis=0)
    recovered_count=final_df['Recovered'].sum(axis=0)
    print('confirmed_count: ',confirmed_count, 'recovered_count: ',recovered_count, 'deceased_count: ',deceased_count)
    return (confirmed_count, recovered_count, deceased_count)

def Q1_2(json_file_path, start_date, end_date):
    
    if(validate(start_date,"%Y-%m-%d")==False):
        return
    if(validate(end_date,"%Y-%m-%d")==False):
        return
    if(greater(start_date,end_date)==False):
        return
    confirmed_count=0
    recovered_count=0
    deceased_count=0
    df,boo=Data_frame(json_file_path,start_date, end_date)
    if(boo==False):
        return
    final_df=get_sorted(df,'dl')
    confirmed_count=final_df['Confirmed'].sum(axis=0)
    deceased_count=final_df['Deceased'].sum(axis=0)
    recovered_count=final_df['Recovered'].sum(axis=0)
    print('confirmed_count: ',confirmed_count, 'recovered_count: ',recovered_count, 'deceased_count: ',deceased_count)
    return (confirmed_count, recovered_count, deceased_count)

def Q1_3(json_file_path, start_date, end_date):
    
    if(validate(start_date,"%Y-%m-%d")==False):
        return
    if(validate(end_date,"%Y-%m-%d")==False):
        return
    if(greater(start_date,end_date)==False):
        return
    confirmed_count=0
    recovered_count=0
    deceased_count=0
    df,boo=Data_frame(json_file_path,start_date, end_date)
    if(boo==False):
        return
    final_df=get_sorted(df,'dl')
    confirmed_count=final_df['Confirmed'].sum(axis=0)
    deceased_count=final_df['Deceased'].sum(axis=0)
    recovered_count=final_df['Recovered'].sum(axis=0)

    final_df=get_sorted(df,'mh')
    confirmed_count+=final_df['Confirmed'].sum(axis=0)
    deceased_count+=final_df['Deceased'].sum(axis=0)
    recovered_count+=final_df['Recovered'].sum(axis=0)
    print('confirmed_count: ',confirmed_count, 'recovered_count: ',recovered_count, 'deceased_count: ',deceased_count)
    return (confirmed_count, recovered_count, deceased_count)

def Q1_4(json_file_path, start_date, end_date):
    
    if(validate(start_date,"%Y-%m-%d")==False):
        return
    if(validate(end_date,"%Y-%m-%d")==False):
        return
    if(greater(start_date,end_date)==False):
        return
    df,boo=Data_frame(json_file_path,start_date, end_date)
    if(boo==False):
        return
    df=get_states(df)
    sum_confirm = df.loc[df['status']=="Confirmed"].sum(axis=0)
    sum_recovered = df.loc[df['status']=="Recovered"].sum(axis=0)
    sum_deceased = df.loc[df['status']=="Deceased"].sum(axis=0)
    print('Confirmed\n')
    lis,lis_conf,final_conf=GetValueMax(sum_confirm)
    # lis=GetValueMax(sum_confirm)[0]
    # lis_conf=GetValueMax(sum_confirm)[1]
    # final_conf=GetValueMax(sum_confirm)[2]
    for i in lis:

        print('Highest affected State is: ',State_Mappings[lis_conf[i]])
        print('Highest affected State count is: ',final_conf[i])
    print('Recovered \n')
    lis,lis_conf,final_conf=GetValueMax(sum_recovered)
    # lis=GetValueMax(sum_recovered)[0]
    # lis_conf=GetValueMax(sum_recovered)[1]
    # final_conf=GetValueMax(sum_recovered)[2]
    for i in lis:
        print('Highest affected State is: ',State_Mappings[lis_conf[i]])
        print('Highest affected State count is: ',final_conf[i])
    print('Deceased\n')
    lis,lis_conf,final_conf=GetValueMax(sum_deceased)
    # lis=GetValueMax(sum_deceased)[0]
    # lis_conf=GetValueMax(sum_deceased)[1]
    # final_conf=GetValueMax(sum_deceased)[2]
    for i in lis:
        print('Highest affected State is: ',State_Mappings[lis_conf[i]])
        print('Highest affected State count is: ',final_conf[i])
def Q1_5(json_file_path, start_date, end_date):
    
    if(validate(start_date,"%Y-%m-%d")==False):
        return
    if(validate(end_date,"%Y-%m-%d")==False):
        return
    if(greater(start_date,end_date)==False):
        return
    df,boo=Data_frame(json_file_path,start_date, end_date)
    if(boo==False):
        return
    df=get_states(df)
    sum_confirm = df.loc[df['status']=="Confirmed"].sum(axis=0)
    sum_recovered = df.loc[df['status']=="Recovered"].sum(axis=0)
    sum_deceased = df.loc[df['status']=="Deceased"].sum(axis=0)
    print('Confirmed \n')
    lis,lis_conf,final_conf=GetValueMin(sum_confirm)
    # lis=GetValueMin(sum_confirm)[0]
    # lis_conf=GetValueMin(sum_confirm)[1]
    # final_conf=GetValueMin(sum_confirm)[2]
    for i in lis:
        print('Lowest affected State is: ',State_Mappings[lis_conf[i]])
        print('Lowest affected State count is: ',final_conf[i])
    print('Recovered \n')
    lis,lis_conf,final_conf=GetValueMin(sum_recovered)
    # lis=GetValueMin(sum_recovered)[0]
    # lis_conf=GetValueMin(sum_recovered)[1]
    # final_conf=GetValueMin(sum_recovered)[2]
    for i in lis:
        print('Lowest affected State is: ',State_Mappings[lis_conf[i]])
        print('Lowest affected State count is: ',final_conf[i])
    print('Deceased \n')
    lis,lis_conf,final_conf=GetValueMin(sum_deceased)
    # lis=GetValueMin(sum_deceased)[0]
    # lis_conf=GetValueMin(sum_deceased)[1]
    # final_conf=GetValueMin(sum_deceased)[2]
    for i in lis:
        print('Lowest affected State is: ',State_Mappings[lis_conf[i]])
        print('Lowest affected State count is: ',final_conf[i])
def Q1_6(json_file_path, start_date, end_date):
    
    if(validate(start_date,"%Y-%m-%d")==False):
        return
    if(validate(end_date,"%Y-%m-%d")==False):
        return
    if(greater(start_date,end_date)==False):
        return
    df,boo=Data_frame(json_file_path,start_date, end_date)
    if(boo==False):
        return
    final_df=get_sorted(df,'dl')

    print('Confirmed \n')
    print('Day: ',final_df['date'][final_df['Confirmed'].idxmax()])
    print('Count: ',final_df['Confirmed'].max())
    print('Recovered \n')
    print('Day: ',final_df['date'][final_df['Recovered'].idxmax()])
    print('Count: ',final_df['Recovered'].max())
    print('Deceased \n')
    print('Day: ',final_df['date'][final_df['Deceased'].idxmax()])
    print('Count: ',final_df['Deceased'].max())


def Q1_7(json_file_path, start_date, end_date):
    
    if(validate(start_date,"%Y-%m-%d")==False):
        return
    if(validate(end_date,"%Y-%m-%d")==False):
        return
    if(greater(start_date,end_date)==False):
        return
    df,boo=Data_frame(json_file_path,start_date, end_date)
    if(boo==False):
        return
    df=get_states(df)
    sum_confirm = df.loc[df['status']=="Confirmed"].sum(axis=0)
    sum_recovered = df.loc[df['status']=="Recovered"].sum(axis=0)
    sum_deceased = df.loc[df['status']=="Deceased"].sum(axis=0)
    final_confirm=sum_confirm.drop(labels=['date', 'status','tt'])
    final_recovered=sum_recovered.drop(labels=['date', 'status','tt'])
    final_deceased=sum_deceased.drop(labels=['date', 'status','tt'])

    tem=final_recovered.add(final_deceased, fill_value=0)
    final_list=final_confirm.subtract(tem, fill_value=0)
    for i in final_list.index:
        print(State_Mappings[i],":",final_list[i])


def Q2_1(json_file_path, start_date, end_date):
    
    if(validate(start_date,"%Y-%m-%d")==False):
        return
    if(validate(end_date,"%Y-%m-%d")==False):
        return
    if(greater(start_date,end_date)==False):
        return
    df,boo=Data_frame(json_file_path,start_date, end_date)
    if(boo==False):
        return
    final_df=get_sorted(df,'tt')

    final_df['Total_Confirmed'] = final_df['Confirmed'].cumsum()
    final_df['Total_Recovered'] = final_df['Recovered'].cumsum()
    final_df['Total_Deceased'] = final_df['Deceased'].cumsum()


    final_df.plot(x='date',y=['Total_Confirmed',  'Total_Recovered' , 'Total_Deceased'],kind='area', stacked=False,figsize=(10.0, 7.0))
    plt.title('COVID-19 Data')
    plt.ylabel('Number of Cases')
    plt.xlabel('Dates')
    plt.savefig('q2_1.png', dpi=300)
    plt.show()

def Q2_2(json_file_path, start_date, end_date):
    
    if(validate(start_date,"%Y-%m-%d")==False):
        return
    if(validate(end_date,"%Y-%m-%d")==False):
        return
    if(greater(start_date,end_date)==False):
        return
    df,boo=Data_frame(json_file_path,start_date, end_date)
    if(boo==False):
        return
    final_df=get_sorted(df,'dl')
    final_df['Total_Confirmed'] = final_df['Confirmed'].cumsum()
    final_df['Total_Recovered'] = final_df['Recovered'].cumsum()
    final_df['Total_Deceased'] = final_df['Deceased'].cumsum()
    final_df.plot(x='date',y=['Total_Confirmed',  'Total_Recovered' , 'Total_Deceased'],kind='area', stacked=False,figsize=(10.0, 7.0))
    plt.title('COVID-19 Data for Delhi')
    plt.ylabel('Number of Cases')
    plt.xlabel('Dates')
    plt.savefig('q2_2.png', dpi=300)
    plt.show()

def Q2_3(json_file_path, start_date, end_date):
    
    if(validate(start_date,"%Y-%m-%d")==False):
        return
    if(validate(end_date,"%Y-%m-%d")==False):
        return
    if(greater(start_date,end_date)==False):
        return
    df,boo=Data_frame(json_file_path,start_date, end_date)
    if(boo==False):
        return
    final_df=get_sorted(df,'tt')
    final_df['Daily_Active']=final_df['Confirmed'] - final_df['Recovered']- final_df['Deceased']
    final_df['Active'] = final_df['Daily_Active'].cumsum()
    final_df.plot(x='date',y='Active',kind='area', stacked=False,figsize=(10.0, 7.0))
    plt.title('Active Number of cases')
    plt.ylabel('Number of Cases')
    plt.xlabel('Dates')
    plt.savefig('q2_3.png', dpi=300)
    plt.show()
def Q3(json_file_path, start_date, end_date):
        
    if(validate(start_date,"%Y-%m-%d")==False):
        return
    if(validate(end_date,"%Y-%m-%d")==False):
        return
    if(greater(start_date,end_date)==False):
        return
    df,boo=Data_frame(json_file_path,start_date, end_date)
    if(boo==False):
        return
    final_df=get_sorted(df,'dl')
    final_df.sort_values(by=['date'],inplace = True)

    final_df["date_modified"] = final_df["date"].astype('category')
    final_df["date_modified"] = final_df["date_modified"].cat.codes
    C_Slope,C_Intercept=LinearRegression(final_df["date_modified"],final_df["Confirmed"],"Confirmed",final_df["date"])
    R_Slope,R_Intercept=LinearRegression(final_df["date_modified"],final_df["Recovered"],"Recovered",final_df["date"])
    D_Slope,D_Intercept=LinearRegression(final_df["date_modified"],final_df["Deceased"],"Deceased",final_df["date"])
    return(C_Intercept,C_Slope,R_Intercept,R_Slope,D_Intercept,D_Slope)

if __name__ == "__main__":
    # execute only if run as a script
    print('2018378 & 2018348') # Please put this first

    # start_date = "2022-03-15"

    # end_date = "2021-04-05"
    start_date = "2020-05-14"
    end_date = "2020-09-05"
    
        
    print(Q1_1('states_daily.json', start_date, end_date)) 
    print(Q1_2('states_daily.json', start_date, end_date)) 
    print(Q1_3('states_daily.json', start_date, end_date)) 
    Q1_4('states_daily.json', start_date, end_date)
    Q1_5('states_daily.json', start_date, end_date)
    Q1_6('states_daily.json', start_date, end_date)
    Q1_7('states_daily.json', start_date, end_date)
    Q2_1('states_daily.json', start_date, end_date)
    Q2_2('states_daily.json', start_date, end_date)
    Q2_3('states_daily.json', start_date, end_date)
    print(Q3('states_daily.json', start_date, end_date))

