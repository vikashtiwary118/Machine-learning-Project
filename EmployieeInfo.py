# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

data=pd.read_csv('Training.csv')

print(data.head(1))
y=data['Response']

cols_to_remove=['Id','BMI','Response']
data1 = data.drop(cols_to_remove,axis=1)

l1=LabelEncoder()
data1['product_Info_2']=l1.fit_transform(data['Product_Info_2'])
data1=data1.drop(['Product_Info_2'],axis=1)

cater_cols=['Product_Info_1','product_Info_2','Product_Info_3','Product_Info_5','Product_Info_6','Product_Info_7','Employment_Info_3','Employment_Info_4','Employment_Info_5','InsuredInfo_1','InsuredInfo_2','InsuredInfo_4','InsuredInfo_5','InsuredInfo_6','InsuredInfo_7','Insurance_History_1','Insurance_History_2','Insurance_History_3','Insurance_History_4','Insurance_History_7','Insurance_History_8','Insurance_History_9','Family_Hist_1','Medical_History_3','Medical_History_4','Medical_History_5','Medical_History_6','Medical_History_7','Medical_History_8','Medical_History_9','Medical_History_11','Medical_History_12','Medical_History_13','Medical_History_14','Medical_History_16','Medical_History_17','Medical_History_18','Medical_History_19','Medical_History_20','Medical_History_21','Medical_History_22','Medical_History_23','Medical_History_25','Medical_History_26','Medical_History_27','Medical_History_28','Medical_History_29','Medical_History_30','Medical_History_31','Medical_History_33','Medical_History_34','Medical_History_35','Medical_History_36','Medical_History_37','Medical_History_38','Medical_History_39','Medical_History_40','Medical_History_41','Medical_Keyword_1','Medical_Keyword_2','Medical_Keyword_3','Medical_Keyword_4','Medical_Keyword_5','Medical_Keyword_6','Medical_Keyword_7','Medical_Keyword_8','Medical_Keyword_9','Medical_Keyword_10','Medical_Keyword_11','Medical_Keyword_15','Medical_Keyword_16','Medical_Keyword_17','Medical_Keyword_20','Medical_Keyword_21','Medical_Keyword_22','Medical_Keyword_23','Medical_Keyword_24','Medical_Keyword_25','Medical_Keyword_26','Medical_Keyword_27','Medical_Keyword_28','Medical_Keyword_29','Medical_Keyword_30','Medical_Keyword_31','Medical_Keyword_32','Medical_Keyword_33','Medical_Keyword_34','Medical_Keyword_35','Medical_Keyword_36','Medical_Keyword_37','Medical_Keyword_38','Medical_Keyword_39','Medical_Keyword_40','Medical_Keyword_41','Medical_Keyword_42','Medical_Keyword_43','Medical_Keyword_44','Medical_Keyword_45','Medical_Keyword_46','Medical_Keyword_47','Medical_Keyword_48']

contin_cols=['Product_Info_4','Ins_Age','Ht','Wt','Medical_Keyword_18','Medical_Keyword_19','Medical_Keyword_12','Medical_Keyword_13','Medical_Keyword_14','Medical_History_32','Medical_History_10','Employment_Info_1','Employment_Info_2','Employment_Info_6','InsuredInfo_3','Insurance_History_5','Family_Hist_2','Family_Hist_3','Family_Hist_4','Family_Hist_5','Medical_History_1','Medical_History_2','Medical_History_15','Medical_History_24']

total_df=cater_cols+contin_cols


cater=pd.DataFrame(data1,columns=cater_cols)
conti=pd.DataFrame(data1,columns=contin_cols)


impca=Imputer(missing_values='NaN',strategy='most_frequent')
impca_out=impca.fit_transform(cater)
impca_df=pd.DataFrame(impca_out,columns=cater_cols)

impco=Imputer(missing_values='NaN',strategy='mean')
impo_out = impco.fit_transform(conti)
impco_df=pd.DataFrame(impo_out,columns=contin_cols)

total_df_data=pd.concat([impca_df,impco_df],axis=1)

sl1=StandardScaler()
sl2=sl1.fit_transform(total_df_data)
sl3=pd.DataFrame(sl2,columns=total_df)

X_train,X_test,Y_train,Y_test=train_test_split(sl3,y,test_size=0.2,random_state=42)
rfe=RandomForestClassifier()
rfe.fit(X_train,Y_train)

y_pred=rfe.predict(X_test)
print(accuracy_score(y_pred,Y_test))

sns.heatmap(data=total_df_data.corr(),annot=True)


data1=total_df_data.corr()
for i in data1:
    for j in data1:
        print(i,'aaaa')
        print(j,'bbbbb')
        if(i==j):
            print(data1[i][j],'same coloumn')
        else:
            print(data1[i][j],'diffrnt')
a=set()      
for i in data1:
    for j in data1 :
        if(i==j):
            continue
        else:
            if(data1[i][j]>0.7):
                a.add(i)
print(a)
                
            
    
    
            
                     
    
        








