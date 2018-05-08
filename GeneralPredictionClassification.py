# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pandas.api.types import is_string_dtype
import sys
import os


def train_test(csv_data,resp):
    location,file=os.path.split(csv_data)
    name,ext=file.split('.')
    if ext=='csv':
        print('Extension checked')
    else:
        print('Wrong input file')
        return 'wrong input file'
    #print("The input file is{}{}".format(filename,fileExtension))
   # if(fileExtension==".csv"):
    #   print("File format correct")
   # else:
    #    print("check the extension of file")

        
    try:
        data=pd.read_csv(csv_data)
    except FileNotFoundError as e:
        return 'File not found'
        
    str_col = []
    non_str_col = []
    if(resp not in data.columns):
        print('response column is not present')
        return 'Nopeeee'
    else:
        pass
    y = data[resp]
    
    remove_col = y
    data1 = data.drop([resp] ,axis = 1)
    
    
    for i in data1.columns[:]:
        if(is_string_dtype(data[i])):
            str_col.append(i)
        else:
            non_str_col.append(i)
    
    
    
    
    
    
    l1 = LabelEncoder()
    for i in str_col:
        print(i)
        data1[i] = l1.fit_transform(data1[i].astype(str))

        
    data_categ = pd.DataFrame(data1,columns = str_col)
    data_conti = pd.DataFrame(data1,columns = non_str_col)
    k=0
    t=0
    if (len(str_col)>0):
        impca=Imputer(missing_values='NaN',strategy='most_frequent')
        impca_out=impca.fit_transform(data_categ)
        impca_df=pd.DataFrame(impca_out,columns=str_col)
        k=1
        
    if (len(non_str_col)>0):
        
        impco=Imputer(missing_values='NaN',strategy='mean')
        impo_out = impco.fit_transform(data_conti)
        impco_df=pd.DataFrame(impo_out,columns=non_str_col)
        t=1
        
    if (k==1 and t==1):
        total_df_data=pd.concat([impca_df,impco_df],axis=1)
    if (k==1 and t==0):
        total_df_data=impca_df
    if (k==0 and t==1):
        total_df_data=impco_df    
        
    
    sl1=StandardScaler()
    sl2=sl1.fit_transform(total_df_data)
    sl3=pd.DataFrame(sl2,columns=total_df_data.columns)    
    
    
    X_train,X_test,Y_train,Y_test=train_test_split(sl3,y,test_size=0.2,random_state=42)
    
    rfe=RandomForestClassifier()
    rfe.fit(X_train,Y_train)
    y_pred=rfe.predict(X_test)
    
    y=accuracy_score(y_pred,Y_test)
    print(y)
    return y


if __name__=='__main__':    
    print( "Take the csv file", sys.argv[1])
    print( "response variable ", sys.argv[2])
    data=sys.argv[1]
   # if(sys.argv[2]!=y):
    #    print('column is not present in table')
    #else:
    resp=sys.argv[2]
    
    rrr=train_test(data,resp)
    print(rrr)



    
    
    

