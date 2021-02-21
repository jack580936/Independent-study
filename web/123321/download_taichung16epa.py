#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import math
import fiona
import folium
import branca.colormap as cm
import requests
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Polygon
import urllib.request
from urllib import request
from shapely.geometry import shape, Point
import os
import threading
from rtree import index
import time
from datetime import datetime
import csv
from keras.models import load_model
import datetime
from sklearn.utils import shuffle


# In[61]:


pr0_time=(datetime.datetime.now()+datetime.timedelta(hours=0)).strftime("%Y-%m-%d %H:00")
pr1_time=(datetime.datetime.now()+datetime.timedelta(hours=+1)).strftime("%Y-%m-%d %H:00")
pr2_time=(datetime.datetime.now()+datetime.timedelta(hours=+2)).strftime("%Y-%m-%d %H:00")
pr3_time=(datetime.datetime.now()+datetime.timedelta(hours=+3)).strftime("%Y-%m-%d %H:00")
pr4_time=(datetime.datetime.now()+datetime.timedelta(hours=+4)).strftime("%Y-%m-%d %H:00")
pr5_time=(datetime.datetime.now()+datetime.timedelta(hours=+5)).strftime("%Y-%m-%d %H:00")
pr6_time=(datetime.datetime.now()+datetime.timedelta(hours=+6)).strftime("%Y-%m-%d %H:00")

#now_time=datetime.datetime.now().strftime("%Y-%m-%d %H:00")
now_time=(datetime.datetime.now()+datetime.timedelta(hours=-1)).strftime("%Y-%m-%d %H:00")

# 2018-05-08 16:54
last12hours=(datetime.datetime.now()+datetime.timedelta(hours=-24)).strftime("%Y-%m-%d %H:00")
now_time,last12hours,pr0_time


# In[3]:


CSV_URL = 'http://aqi.thu.edu.tw/echarts/getairquality?start='+last12hours+'&end='+now_time+'&site=西屯,忠明,豐原,沙鹿,大里,后里,太平,大甲,霧峰,烏日,文山,交通測站,梧棲,清水,大肚,東大,龍井,交通監測車'

r = requests.get(CSV_URL)
with open('/home/gh555657/123321/taichung_all_epa.csv', 'wb') as f:
    f.write(r.content)


# In[4]:


taichung16= pd.read_csv(u"/home/gh555657/123321/taichung_all_epa.csv",encoding='big5')
taichung16=taichung16[['SiteName','County','Area','latitude','longitude','AQI','SO2','CO','O3','PM10','PM25','NO2','NO','datetime']]
taichung16.replace({0:np.nan,0.0:np.nan}, inplace=True)
taichung16.replace('ND',np.nan, inplace=True)
taichung16


# In[5]:


'''
model_16name = [
    'model_xitun', 'model_thu', 'model_wenshan', 'model_CM', 'model_taiping',
    'model_dali', 'model_dadu', 'model_dajia', 'model_wuri', 'model_shalu',
    'model_longjing', 'model_wufeng', 'model_qingshui', 'model_fengyuan',
    'model_wuqi', 'model_tc', 'model_houli'
]
model_16filename = [
    '/home/gh555657/epa_16_predict_model/Xitunmodel.h5',
    '/home/gh555657/epa_16_predict_model/THUmodel.h5',
    '/home/gh555657/epa_16_predict_model/Wenshanmodel2.h5',
    '/home/gh555657/epa_16_predict_model/CMmodel.h5',
    '/home/gh555657/epa_16_predict_model/Taipingmodel.h5',
    '/home/gh555657/epa_16_predict_model/Dalimodel3.h5',
    '/home/gh555657/epa_16_predict_model/Dadumodel.h5',
    '/home/gh555657/epa_16_predict_model/Dajiamodel.h5',
    '/home/gh555657/epa_16_predict_model/Wurimodel.h5',
    '/home/gh555657/epa_16_predict_model/Shalumodel.h5',
    '/home/gh555657/epa_16_predict_model/Longjingmodel.h5',
    '/home/gh555657/epa_16_predict_model/Wufengmodel.h5',
    '/home/gh555657/epa_16_predict_model/Qingshuimodel.h5',
    '/home/gh555657/epa_16_predict_model/Fengyuanmodel.h5',
    '/home/gh555657/epa_16_predict_model/Wuqimodel.h5',
    '/home/gh555657/epa_16_predict_model/Tcmodel2.h5',
    '/home/gh555657/epa_16_predict_model/Houlimodel.h5'
]

# --------model_load--------


def getmodel(modelname):
    from keras.models import load_model
    global model_xitun, model_thu, model_wenshan, model_CM, model_taiping, model_dali, model_dadu, model_dajia, model_wuri
    global model_shalu, model_longjing, model_wufeng, model_qingshui, model_fengyuan, model_wuqi, model_tc, model_houli
    if (modelname == 0):  # 讀取西屯模型
        model_xitun = load_model('/home/gh555657/epa_16_predict_model/Xitunmodel.h5')
        return model_xitun 
    if (modelname == 1):  # 讀取東大模型
        model_thu = load_model('/home/gh555657/epa_16_predict_model/THUmodel.h5')
        return model_thu 
    if (modelname == 2):  # 讀取文山模型
        model_wenshan = load_model('/home/gh555657/epa_16_predict_model/Wenshanmodel2.h5')
        return model_wenshan
    if (modelname == 3):  # 讀取忠明模型
        model_CM = load_model('/home/gh555657/epa_16_predict_model/CMmodel.h5')
        return model_CM
    if (modelname == 4):  # 讀取太平模型
        model_taiping = load_model('/home/gh555657/epa_16_predict_model/Taipingmodel.h5')
        return model_taiping
    if (modelname == 5):  # 讀取大里模型
        model_dali = load_model('/home/gh555657/epa_16_predict_model/Dalimodel3.h5')
        return model_dali
    if (modelname == 6):  # 讀取大肚模型
        model_dadu = load_model('/home/gh555657/epa_16_predict_model/Dadumodel.h5')
        return model_dadu
    if (modelname == 7):  # 讀取大甲模型
        model_dajia = load_model('/home/gh555657/epa_16_predict_model/Dajiamodel.h5')
        return model_dajia
    if (modelname == 8):  # 讀取烏日模型
        model_wuri = load_model('/home/gh555657/epa_16_predict_model/Wurimodel.h5')
        return model_wuri
    if (modelname == 9):  # 讀取沙鹿模型
        model_shalu = load_model('/home/gh555657/epa_16_predict_model/Shalumodel.h5')
        return model_shalu
    if (modelname == 10):  # 讀取龍井模型
        model_longjing = load_model('/home/gh555657/epa_16_predict_model/Longjingmodel.h5')
        return model_longjing
    if (modelname == 11):  # 讀取霧峰模型
        model_wufeng = load_model('/home/gh555657/epa_16_predict_model/Wufengmodel.h5')
        return model_wufeng
    if (modelname == 12):  # 讀取清水模型
        model_qingshui = load_model('/home/gh555657/epa_16_predict_model/Qingshuimodel.h5')
        return model_qingshui
    if (modelname == 13):  # 讀取豐原模型
        model_fengyuan = load_model('/home/gh555657/epa_16_predict_model/Fengyuanmodel.h5')
        return model_fengyuan
    if (modelname == 14):  # 讀取梧棲模型
        model_wuqi = load_model('/home/gh555657/epa_16_predict_model/Wuqimodel.h5')
        return model_wuqi
    if (modelname == 15):  # 讀取交通監測車模型
        model_tc = load_model('/home/gh555657/epa_16_predict_model/Tcmodel2.h5')
        return model_tc
    if (modelname == 16):  # 讀取后里模型
        model_houli = load_model('/home/gh555657/epa_16_predict_model/Houlimodel.h5')
        return model_houli


# -------------------多執行緒-----------


def get_modelnamebythread(totaldata):
    totalnum = totaldata  # 總執行次數
    Q = int(totalnum / 5)  # 取商數
    R = totalnum % 5  # 取餘數

    for i in range(Q):
        threads = []
        for j in range(5):  # 開多少執行緒
            threads.append(
                threading.Thread(
                    target=getmodel,  # 要執行函數
                    args=(i * 5 + j, )))  # 要執行函數的參數
            threads[j].start()

        for j in threads:
            j.join()

    threads = []
    for i in range(R):
        threads.append(threading.Thread(target=getmodel, args=(Q * 5 + i, )))
        threads[i].start()
    for j in threads:
        j.join()
    print("100%")  # 顯示進度


# -------------------取得model----------
get_modelnamebythread(16)
'''


# In[6]:


#--------model_load--------
#讀取西屯模型
model_xitun = load_model('/home/gh555657/epa_16_predict_model/Xitunmodel.h5')


# In[7]:


#讀取東大模型
model_thu = load_model('/home/gh555657/epa_16_predict_model/THUmodel.h5')


# In[8]:


#讀取文山模型
model_wenshan= load_model('/home/gh555657/epa_16_predict_model/Wenshanmodel1.h5')


# In[9]:


#讀取忠明模型
model_CM = load_model('/home/gh555657/epa_16_predict_model/CMmodel1.h5')


# In[10]:


#讀取太平模型
model_taiping = load_model('/home/gh555657/epa_16_predict_model/Taipingmodel.h5')


# In[11]:


#讀取大里模型
model_dali = load_model('/home/gh555657/epa_16_predict_model/Dalimodel2.h5')


# In[12]:


#讀取大肚模型
model_dadu = load_model('/home/gh555657/epa_16_predict_model/Dadumodel.h5')


# In[13]:


#讀取大甲模型
model_dajia = load_model('/home/gh555657/epa_16_predict_model/Dajiamodel.h5')


# In[14]:


#讀取烏日模型
model_wuri = load_model('/home/gh555657/epa_16_predict_model/Wurimodel.h5')


# In[15]:


#讀取沙鹿模型
model_shalu = load_model('/home/gh555657/epa_16_predict_model/Shalumodel.h5')


# In[16]:


#讀取龍井模型
model_longjing = load_model('/home/gh555657/epa_16_predict_model/Longjingmodel.h5')


# In[17]:


#讀取霧峰模型
model_wufeng = load_model('/home/gh555657/epa_16_predict_model/Wufengmodel.h5')


# In[18]:


#讀取清水模型
model_qingshui = load_model('/home/gh555657/epa_16_predict_model/Qingshuimodel.h5')


# In[19]:


#讀取豐原模型
model_fengyuan = load_model('/home/gh555657/epa_16_predict_model/Fengyuanmodel.h5')


# In[20]:


#讀取梧棲模型
model_wuqi = load_model('/home/gh555657/epa_16_predict_model/Wuqimodel.h5')


# In[21]:


#讀取交通監測車模型
model_tc = load_model('/home/gh555657/epa_16_predict_model/Tcmodel2.h5')


# In[22]:


# 讀取后里模型
        model_houli = load_model('/home/gh555657/epa_16_predict_model/Houlimodel.h5')


# In[23]:


#--------西屯6小時----------
Xitun = taichung16['SiteName'].isin(['西屯']) #找西屯測站
Xitun=taichung16[Xitun]

where_null=Xitun.isnull().any(axis=1).reset_index(drop=True)

if(where_null[0]==True):
    Xitun_p=Xitun.fillna(method='bfill').drop(columns=['SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Xitun_p=Xitun_p.fillna(method='ffill')
else:
    Xitun_p=Xitun.fillna(method='ffill').drop(columns=['SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Xitun_p=Xitun_p.fillna(method='bfill')

Xitun_p.reset_index(drop=True,inplace=True)
Xitun_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)
#------正規化---------------↓
from sklearn.preprocessing import StandardScaler

scale = StandardScaler() #z-scaler物件
Xitun_scaled = pd.DataFrame(scale.fit_transform(Xitun_p),columns=Xitun_p.keys())

Xitun_scaled=np.array(Xitun_scaled)
Xitun_scaled = np.reshape(Xitun_scaled, (1, Xitun_scaled.shape[0], Xitun_scaled.shape[1])) 



# In[24]:


#--------東大6小時----------
THU = taichung16['SiteName'].isin(['東大']) #找東大測站
THU=taichung16[THU]
THU=THU.drop(columns=['CO','NO']).reset_index(drop=True)
where_THU_null=THU.isnull().any(axis=1)
if(where_THU_null[0]==True):
    THU_p=THU.fillna(method='bfill').drop(columns=['SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    THU_p=THU_p.fillna(method='ffill')
else:
    THU_p=THU.fillna(method='ffill').drop(columns=['SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    THU_p=THU_p.fillna(method='bfill')

THU_p.reset_index(drop=True,inplace=True)
THU_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)
#------正規化---------------↓


scale = StandardScaler() #z-scaler物件
THU_scaled = pd.DataFrame(scale.fit_transform(THU_p),columns=THU_p.keys())

THU_scaled=np.array(THU_scaled)
THU_scaled = np.reshape(THU_scaled, (1, THU_scaled.shape[0], THU_scaled.shape[1]))


# In[25]:


#--------文山6小時----------
Wenshan = taichung16['SiteName'].isin(['文山']) #找文山測站
Wenshan=taichung16[Wenshan]
Wenshan=Wenshan.drop(columns=['NO']).reset_index(drop=True)
where_null=Wenshan.isnull().any(axis=1)
if(where_null[0]==True):
    Wenshan_p=Wenshan.fillna(method='bfill').drop(columns=['SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Wenshan_p=Wenshan_p.fillna(method='ffill')
else:
    Wenshan_p=Wenshan.fillna(method='ffill').drop(columns=['SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Wenshan_p=Wenshan_p.fillna(method='bfill')

Wenshan_p.reset_index(drop=True,inplace=True)
Wenshan_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)

#------正規化---------------↓


scale = StandardScaler() #z-scaler物件
Wenshan_scaled = pd.DataFrame(scale.fit_transform(Wenshan_p),columns=Wenshan_p.keys())

Wenshan_scaled=np.array(Wenshan_scaled)
Wenshan_scaled = np.reshape(Wenshan_scaled, (1, Wenshan_scaled.shape[0], Wenshan_scaled.shape[1])) 


# In[26]:


#--------忠明6小時----------
CM = taichung16['SiteName'].isin(['忠明']) #找忠明測站
CM=taichung16[CM]

CM=CM.drop(columns=['NO']).reset_index(drop=True)

where_null=CM.isnull().any(axis=1)
if(where_null[0]==True):
    CM_p=CM.fillna(method='bfill').drop(columns=['SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    CM_p=CM_p.fillna(method='ffill')
else:
    CM_p=CM.fillna(method='ffill').drop(columns=['SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    CM_p=CM_p.fillna(method='bfill')
    
#CM_p.drop([0],inplace=True)
CM_p.reset_index(drop=True,inplace=True)

#------正規化---------------↓


scale = StandardScaler() #z-scaler物件
CM_scaled = pd.DataFrame(scale.fit_transform(CM_p),columns=CM_p.keys())

CM_scaled=np.array(CM_scaled)
CM_scaled = np.reshape(CM_scaled, (1, CM_scaled.shape[0], CM_scaled.shape[1])) 


# In[27]:


#--------太平6小時----------
Taiping = taichung16['SiteName'].isin(['太平']) #找太平測站
Taiping=taichung16[Taiping]

Taiping=Taiping.drop(columns=['NO']).reset_index(drop=True)
where_null=Taiping.isnull().any(axis=1)
if(where_null[0]==True):
    Taiping_p=Taiping.fillna(method='bfill').drop(columns=['SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Taiping_p=Taiping_p.fillna(method='ffill')
else:
    Taiping_p=Taiping.fillna(method='ffill').drop(columns=['SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Taiping_p=Taiping_p.fillna(method='bfill')

Taiping_p.reset_index(drop=True,inplace=True)
Taiping_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)

#------正規化---------------↓

scale = StandardScaler() #z-scaler物件
Taiping_scaled = pd.DataFrame(scale.fit_transform(Taiping_p),columns=Taiping_p.keys())

Taiping_scaled=np.array(Taiping_scaled)
Taiping_scaled = np.reshape(Taiping_scaled, (1, Taiping_scaled.shape[0], Taiping_scaled.shape[1])) 


# In[28]:


#--------大里6小時----------
Dali = taichung16['SiteName'].isin(['大里']) #找大里測站
Dali=taichung16[Dali]
where_Dali_null=Dali.isnull().any(axis=1).reset_index(drop=True)
if(where_Dali_null[0]==True):
    Dali_p=Dali.fillna(method='bfill').drop(columns=['SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Dali_p=Dali_p.fillna(method='ffill')
else:
    Dali_p=Dali.fillna(method='ffill').drop(columns=['SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Dali_p=Dali_p.fillna(method='bfill')

#Dali_p.drop([0],inplace=True)
Dali_p.reset_index(drop=True,inplace=True)
#Dali_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)

#------正規化---------------↓


scale = StandardScaler() #z-scaler物件
Dali_scaled = pd.DataFrame(scale.fit_transform(Dali_p),columns=Dali_p.keys())

Dali_scaled=np.array(Dali_scaled)
Dali_scaled = np.reshape(Dali_scaled, (1, Dali_scaled.shape[0], Dali_scaled.shape[1])) 


# In[29]:


#--------大肚6小時----------
Dadu = taichung16['SiteName'].isin(['大肚']) #找大肚測站
Dadu=taichung16[Dadu]
where_null=Dadu.isnull().any(axis=1).reset_index(drop=True)
if(where_null[0]==True):
    Dadu_p=Dadu.fillna(method='bfill').drop(columns=['NO','CO','O3','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Dadu_p=Dadu_p.fillna(method='ffill')
else:
    Dadu_p=Dadu.fillna(method='ffill').drop(columns=['NO','CO','O3','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Dadu_p=Dadu_p.fillna(method='bfill')


Dadu_p.reset_index(drop=True,inplace=True)
Dadu_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)

#------正規化---------------↓


scale = StandardScaler() #z-scaler物件
Dadu_scaled = pd.DataFrame(scale.fit_transform(Dadu_p),columns=Dadu_p.keys())

Dadu_scaled=np.array(Dadu_scaled)
Dadu_scaled = np.reshape(Dadu_scaled, (1, Dadu_scaled.shape[0], Dadu_scaled.shape[1])) 


# In[30]:


#--------大甲6小時----------
Dajia = taichung16['SiteName'].isin(['大甲']) #找大甲測站
Dajia=taichung16[Dajia]
where_null=Dajia.isnull().any(axis=1).reset_index(drop=True)
if(where_null[0]==True):
    Dajia_p=Dajia.fillna(method='bfill').drop(columns=['NO','CO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Dajia_p=Dajia_p.fillna(method='ffill')
else:
    Dajia_p=Dajia.fillna(method='ffill').drop(columns=['NO','CO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Dajia_p=Dajia_p.fillna(method='bfill')


Dajia_p.reset_index(drop=True,inplace=True)
Dajia_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)

#------正規化---------------↓


scale = StandardScaler() #z-scaler物件
Dajia_scaled = pd.DataFrame(scale.fit_transform(Dajia_p),columns=Dajia_p.keys())

Dajia_scaled=np.array(Dajia_scaled)
Dajia_scaled = np.reshape(Dajia_scaled, (1, Dajia_scaled.shape[0], Dajia_scaled.shape[1])) 


# In[31]:


#--------烏日6小時----------
Wuri = taichung16['SiteName'].isin(['烏日']) #找烏日測站
Wuri=taichung16[Wuri]
where_null=Wuri.isnull().any(axis=1).reset_index(drop=True)
if(where_null[0]==True):
    Wuri_p=Wuri.fillna(method='bfill').drop(columns=['NO','O3','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Wuri_p=Wuri_p.fillna(method='ffill')
else:
    Wuri_p=Wuri.fillna(method='ffill').drop(columns=['NO','O3','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Wuri_p=Wuri_p.fillna(method='bfill')
Wuri_p.reset_index(drop=True,inplace=True)
Wuri_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)

#------正規化---------------↓


scale = StandardScaler() #z-scaler物件
Wuri_scaled = pd.DataFrame(scale.fit_transform(Wuri_p),columns=Wuri_p.keys())

Wuri_scaled=np.array(Wuri_scaled)
Wuri_scaled = np.reshape(Wuri_scaled, (1, Wuri_scaled.shape[0], Wuri_scaled.shape[1])) 


# In[32]:


#--------沙鹿6小時----------
Shalu = taichung16['SiteName'].isin(['沙鹿']) #找沙鹿測站
Shalu=taichung16[Shalu]
where_null=Shalu.isnull().any(axis=1).reset_index(drop=True)
if(where_null[0]==True):
    Shalu_p=Shalu.fillna(method='bfill').drop(columns=['NO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Shalu_p=Shalu_p.fillna(method='ffill')
else:
    Shalu_p=Shalu.fillna(method='ffill').drop(columns=['NO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Shalu_p=Shalu_p.fillna(method='bfill')

#Shalu_p.drop([0],inplace=True)
Shalu_p.reset_index(drop=True,inplace=True)
Shalu_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)

#------正規化---------------↓


scale = StandardScaler() #z-scaler物件
Shalu_scaled = pd.DataFrame(scale.fit_transform(Shalu_p),columns=Shalu_p.keys())

Shalu_scaled=np.array(Shalu_scaled)
Shalu_scaled = np.reshape(Shalu_scaled, (1, Shalu_scaled.shape[0], Shalu_scaled.shape[1])) 


# In[33]:


#--------龍井6小時----------
Longjing = taichung16['SiteName'].isin(['龍井']) #找龍井測站
Longjing=taichung16[Longjing]
where_null=Longjing.isnull().any(axis=1).reset_index(drop=True)
if(where_null[0]==True):
    Longjing_p=Longjing.fillna(method='bfill').drop(columns=['NO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Longjing_p=Longjing_p.fillna(method='ffill')
else:
    Longjing_p=Longjing.fillna(method='ffill').drop(columns=['NO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Longjing_p=Longjing_p.fillna(method='bfill')



Longjing_p.reset_index(drop=True,inplace=True)
Longjing_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)

#------正規化---------------↓

scale = StandardScaler() #z-scaler物件
Longjing_scaled = pd.DataFrame(scale.fit_transform(Longjing_p),columns=Longjing_p.keys())

Longjing_scaled=np.array(Longjing_scaled)
Longjing_scaled = np.reshape(Longjing_scaled, (1, Longjing_scaled.shape[0], Longjing_scaled.shape[1])) 


# In[34]:


#--------霧峰6小時----------
Wufeng = taichung16['SiteName'].isin(['霧峰']) #找霧峰測站
Wufeng=taichung16[Wufeng]
where_null=Wufeng.isnull().any(axis=1).reset_index(drop=True)
Wufeng.reset_index(drop=True,inplace=True)
Wufeng.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)
if(where_null[0]==True):
    Wufeng_p=Wufeng.fillna(method='ffill').drop(columns=['NO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Wufeng_p=Wenshan_p.fillna(method='bfill')
else:
    Wufeng_p=Wufeng.fillna(method='bfill').drop(columns=['NO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Wufeng_p=Wenshan_p.fillna(method='ffill')


#------正規化---------------↓


scale = StandardScaler() #z-scaler物件
Wufeng_scaled = pd.DataFrame(scale.fit_transform(Wufeng_p),columns=Wufeng_p.keys())

Wufeng_scaled=np.array(Wufeng_scaled)
Wufeng_scaled = np.reshape(Wufeng_scaled, (1, Wufeng_scaled.shape[0], Wufeng_scaled.shape[1])) 


# In[35]:


Wufeng_p


# In[ ]:





# In[36]:


#--------清水6小時----------
Qingshui = taichung16['SiteName'].isin(['清水']) #找清水測站
Qingshui=taichung16[Qingshui]
where_null=Qingshui.isnull().any(axis=1).reset_index(drop=True)
if(where_null[0]==True):
    Qingshui_p=Qingshui.fillna(method='bfill').drop(columns=['NO','CO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Qingshui_p=Qingshui_p.fillna(method='ffill')
else:
    Qingshui_p=Qingshui.fillna(method='ffill').drop(columns=['NO','CO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Qingshui_p=Qingshui_p.fillna(method='bfill')

Qingshui_p.reset_index(drop=True,inplace=True)
#Qingshui_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)

#------正規化---------------↓


scale = StandardScaler() #z-scaler物件
Qingshui_scaled = pd.DataFrame(scale.fit_transform(Qingshui_p),columns=Qingshui_p.keys())

Qingshui_scaled=np.array(Qingshui_scaled)
Qingshui_scaled = np.reshape(Qingshui_scaled, (1, Qingshui_scaled.shape[0], Qingshui_scaled.shape[1])) 


# In[ ]:





# In[37]:


#--------豐原6小時----------
Fengyuan = taichung16['SiteName'].isin(['豐原']) #找豐原測站
Fengyuan=taichung16[Fengyuan]
where_null=Fengyuan.isnull().any(axis=1).reset_index(drop=True)
if(where_null[0]==True):
    Fengyuan_p=Fengyuan.fillna(method='bfill').drop(columns=['NO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Fengyuan_p=Fengyuan_p.fillna(method='ffill')
else:
    Fengyuan_p=Fengyuan.fillna(method='ffill').drop(columns=['NO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Fengyuan_p=Fengyuan_p.fillna(method='bfill')

#Fengyuan_p.drop([0],inplace=True)
Fengyuan_p.reset_index(drop=True,inplace=True)
#Fengyuan_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)

#------正規化---------------↓


scale = StandardScaler() #z-scaler物件
Fengyuan_scaled = pd.DataFrame(scale.fit_transform(Fengyuan_p),columns=Fengyuan_p.keys())

Fengyuan_scaled=np.array(Fengyuan_scaled)
Fengyuan_scaled = np.reshape(Fengyuan_scaled, (1, Fengyuan_scaled.shape[0], Fengyuan_scaled.shape[1])) 


# In[38]:


#--------梧棲6小時----------
Wuqi = taichung16['SiteName'].isin(['梧棲']) #找梧棲測站
Wuqi=taichung16[Wuqi]
where_null=Wuqi.isnull().any(axis=1).reset_index(drop=True)
if(where_null[0]==True):
    Wuqi_p=Wuqi.fillna(method='bfill').drop(columns=['NO','CO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Wuqi_p=Wuqi_p.fillna(method='ffill')
else:
    Wuqi_p=Wuqi.fillna(method='ffill').drop(columns=['NO','CO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Wuqi_p=Wuqi_p.fillna(method='bfill')


Wuqi_p.reset_index(drop=True,inplace=True)
Wuqi_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)

#------正規化---------------↓


scale = StandardScaler() #z-scaler物件
Wuqi_scaled = pd.DataFrame(scale.fit_transform(Wuqi_p),columns=Wuqi_p.keys())

Wuqi_scaled=np.array(Wuqi_scaled)
Wuqi_scaled = np.reshape(Wuqi_scaled, (1, Wuqi_scaled.shape[0], Wuqi_scaled.shape[1])) 


# In[39]:


#--------交通監測車6小時----------
Tc = taichung16['SiteName'].isin(['交通監測車']) #找交通監測車測站
Tc=taichung16[Tc]
where_null=Tc.isnull().any(axis=1).reset_index(drop=True)
if(where_null[0]==True):
    Tc_p=Tc.fillna(method='bfill').drop(columns=['NO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Tc_p=Tc_p.fillna(method='ffill')
else:
    Tc_p=Tc.fillna(method='ffill').drop(columns=['NO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Tc_p=Tc_p.fillna(method='bfill')
    
#Tc_p.drop([0],inplace=True)
Tc_p.reset_index(drop=True,inplace=True)
Tc_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)

#------正規化---------------↓


scale = StandardScaler() #z-scaler物件
Tc_scaled = pd.DataFrame(scale.fit_transform(Tc_p),columns=Tc_p.keys())

Tc_scaled=np.array(Tc_scaled)
Tc_scaled = np.reshape(Tc_scaled, (1, Tc_scaled.shape[0], Tc_scaled.shape[1])) 


# In[40]:


# --------后里6小時----------
from sklearn.preprocessing import StandardScaler
Houli = taichung16['SiteName'].isin(['后里'])  # 找后里測站
Houli = taichung16[Houli]
where_null=Houli.isnull().any(axis=1).reset_index(drop=True)
if(where_null[0]==True):
    Houli_p=Houli.fillna(method='bfill').drop(columns=['NO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Houli_p=Houli_p.fillna(method='ffill')
else:
    Houli_p=Houli.fillna(method='ffill').drop(columns=['NO','SiteName','AQI','Area','County','latitude','longitude','datetime']).reset_index(drop=True)
    Houli_p=Houli_p.fillna(method='bfill')

Houli_p.reset_index(drop=True, inplace=True)
Houli_p.drop([12,13,14,15,16,17,18,19,20,21,22,23],inplace=True)

# ------正規化---------------↓

scale = StandardScaler()  # z-scaler物件
Houli_scaled = pd.DataFrame(
    scale.fit_transform(Houli_p), columns=Houli_p.keys())

Houli_scaled = np.array(Houli_scaled)
Houli_scaled = np.reshape(
    Houli_scaled, (1, Houli_scaled.shape[0], Houli_scaled.shape[1]))


# In[ ]:





# In[41]:


# 預測西屯PM25

Xitunpredict = model_xitun.predict(Xitun_scaled)
# 預測東大PM25

THUpredict = model_thu.predict(THU_scaled)
# 預測文山PM25

Wenshanpredict = model_wenshan.predict(Wenshan_scaled)
# 預測忠明PM25

CMpredict = model_CM.predict(CM_scaled)
# 預測太平PM25

Taipingpredict = model_taiping.predict(Taiping_scaled)
# 預測大里PM25

Dalipredict = model_dali.predict(Dali_scaled)
# 預測大肚PM25

Dadupredict = model_dadu.predict(Dadu_scaled)
# 預測大甲PM25

Dajiapredict = model_dajia.predict(Dajia_scaled)
# 預測烏日PM25

Wuripredict = model_wuri.predict(Wuri_scaled)
# 預測沙鹿PM25

Shalupredict = model_shalu.predict(Shalu_scaled)
# 預測龍井PM25

Longjingpredict = model_longjing.predict(Longjing_scaled)
# 預測霧峰PM25

Wufengpredict = model_wufeng.predict(Wufeng_scaled)
# 預測清水PM25

Qingshuipredict = model_qingshui.predict(Qingshui_scaled)
# 預測豐原PM25

Fengyuanpredict = model_fengyuan.predict(Fengyuan_scaled)
# 預測梧棲PM25

Wuqipredict = model_wuqi.predict(Wuqi_scaled)
# 預測交通監測車PM25

Tcpredict = model_tc.predict(Tc_scaled)
# 預測后里PM25

Houlipredict = model_houli.predict(Houli_scaled)


# In[42]:


a=[Xitunpredict, CMpredict, Dalipredict, Shalupredict, Fengyuanpredict, Wenshanpredict, Dajiapredict, Taipingpredict, Wufengpredict, Wuripredict, Houlipredict, Wuqipredict, Dadupredict,THUpredict, Qingshuipredict, Longjingpredict]


# In[43]:


#--------------地圖預測---------------
import json
import math
import fiona
import folium
import branca.colormap as cm
import requests
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Polygon
import urllib.request
from urllib import request
from shapely.geometry import shape, Point
import os
import threading
from rtree import index
import time
import csv 


# In[44]:


prepm =pd.DataFrame(a[1])
prepm25= pd.DataFrame(a[0])
for i in range(15):
    prepm =pd.DataFrame(a[i+1])
    prepm25= prepm25.append(prepm)
prepm25


# In[45]:


prepm25.columns=['0hr','1hr','2hr','3hr','4hr','5hr','6hr']
prepm25.index= range(0,len(prepm25))
'''
prevalue = pd.read_csv("/home/gh555657/123321/predict_value.csv")
prevalue=prevalue.append(prepm25).reset_index(drop=True)
prevalue.to_csv("/home/gh555657/123321/predict_value.csv")
prevalue=prepm25.to_csv('/home/gh555657/123321/predict_value.csv', mode='a', header=False)
'''


# In[62]:


#df.rename({1: 2, 2: 4}, axis='index'
prepm25_save=prepm25
prepm25_save=prepm25_save.T
prepm25_save.rename({'0hr':pr0_time,'1hr':pr1_time,'2hr':pr2_time,'3hr':pr3_time,'4hr':pr4_time
                     ,'5hr':pr5_time,'6hr':pr6_time},axis='index',inplace=True)
prepm25_save.to_csv('/home/gh555657/123321/predict_value.csv', mode='a', header=False)
prepm25_save


# In[63]:


#========================================================================================================
#爬蟲 (環保署測站、時間、風力資訊、台中各區天氣) + idw

ses = requests.Session()
data1 = ses.get('http://taqm.epb.taichung.gov.tw/TQAMNEWAQITABLE.ASPX') #環保署16筆測站
data1.encoding = 'utf-8'
t = pd.read_html(data1.text)[0]
t.drop(t.iloc[:, 1:21], inplace=True, axis=0)
t.drop(t.iloc[:, 1:279], inplace=True, axis=1)
times = str(t[0])


# In[64]:


# print(times[21:36]) 時間
df1 = pd.read_html(data1.text)[1]
cols1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
df1 = df1.replace('─','0')
df1 = df1.drop(df1.columns[cols1],axis=1)
df1.rename(columns={ df1.columns[0]: "SiteName"}, inplace=True)
df1.rename(columns={ df1.columns[1]: "PM2.5"}, inplace=True)
df1['Latitude']=[24.1622,24.151958,24.099611,24.225628,
                 24.256586,24.139008,24.350426,24.139564,
                 24.05735,24.094264,24.307036,24.250388,
                 24.150919,24.182055,24.269233,24.20103]

df1['Longitude']=[120.616917,120.641092,120.677689,120.568794,
                  120.741711,120.597876,120.615358,120.715064,
                  120.697299,120.646629,120.714881,120.538839,
                  120.540877,120.60707,120.576421,120.498566]
df1.rename(columns={"測站名稱":"SiteName"}, inplace=True)
df1.rename(columns={ "細懸浮微粒(PM2.5)":"PM2.5"}, inplace=True)


df1


# In[65]:


df1[['Latitude', 'Longitude', 'PM2.5']] = df1[['Latitude', 'Longitude','PM2.5']].astype(float)
#df1  顯示16筆測站
df1.to_csv("/home/gh555657/123321/df1.csv")
df1 = pd.read_csv("/home/gh555657/123321/df1.csv")


# In[66]:


df1.drop([0],inplace=True) #把df1不要的第一row砍掉
df11=df1['SiteName']       #先測站把名稱存進df11


df1.drop(["SiteName"],axis=1,inplace=True)#測站名稱column砍掉
df1.rename(columns={ 'Unnamed: 0':'Id'}, inplace=True)#更改unname--->Id
#df1=df1[['Id','SiteName','PM2.5','Latitude','Longitude']] df1目前型態
df1=df1.astype('float64')#轉成float64L
df11.index= range(0,len(df11))#index重排
df1.index=range(0,len(df1))


# In[67]:


#定義IDW

def idw(data_obs,data_grid,radius,interval):
    print('半徑 %.2Fkm 範圍內無觀測站則無法觀測'%(radius+interval))
    df_goal = pd.DataFrame([])
    for grid_i in range(len(data_grid)):
        data_obs['grid_ID'] = data_grid.loc[grid_i,'grid_ID']
        df_work = pd.merge(data_grid,data_obs,how='inner',on='grid_ID')
    
        size= 0.0092
    
        df_work['Lat_dis_km']   = abs((df_work['Latitude']  - df_work['grid_Latitude']) /size)
        df_work['Lon_dis_km']   = abs((df_work['Longitude'] - df_work['grid_Longitude'])/size)
        df_work['distance']     = np.sqrt(df_work['Lat_dis_km']**2+df_work['Lon_dis_km']**2)
        df_work['distance_rec'] = 1/df_work['distance']

        df_work = df_work.sort_values(by=['distance'])
        df_work = df_work.reset_index()

 
        #做出圓心
        center_list = df_work.loc[df_work['distance']<=radius]
        df_center = pd.DataFrame([])
        if len(center_list)==0:
            df_center = pd.DataFrame({'grid_ID':[grid_i+1],'center_PM2.5':[None]})
        else:
            center_temp = center_list.groupby('grid_ID')['distance_rec'].sum()
            center_list = pd.merge(center_list,center_temp,how='inner',on='grid_ID')
            center_list['center_PM2.5']=(center_list['distance_rec_x']/center_list['distance_rec_y'])*center_list['PM2.5']
            df_center = pd.DataFrame(center_list.groupby(['grid_ID'])['center_PM2.5'].sum()).reset_index()  
    
        #做出第1層圓環
        L1_list = df_work.loc[(df_work['distance']>radius)&(df_work['distance']<=(radius+interval))]
        df_L1 = pd.DataFrame([])
        if len(L1_list)==0:
            df_L1 = pd.DataFrame({'grid_ID':[grid_i+1],'L1_PM2.5':[None]})
        else:
            L1_temp = L1_list.groupby('grid_ID')['distance_rec'].sum()
            L1_list = pd.merge(L1_list,L1_temp,how='inner',on='grid_ID')
            L1_list['L1_PM2.5']=(L1_list['distance_rec_x']/L1_list['distance_rec_y'])*L1_list['PM2.5']
            df_L1 = pd.DataFrame(L1_list.groupby(['grid_ID'])['L1_PM2.5'].sum()).reset_index()
 
        #做出第2層圓環
        L2_list = df_work.loc[(df_work['distance']>(radius+interval))&(df_work['distance']<=(radius+interval*2))]
        df_L2 = pd.DataFrame([])
        if len(L2_list)==0:
             df_L2 = pd.DataFrame({'grid_ID':[grid_i+1],'L2_PM2.5':[None]})
        else:
            L2_temp = L2_list.groupby('grid_ID')['distance_rec'].sum()
            L2_list = pd.merge(L2_list,L2_temp,how='inner',on='grid_ID')
            L2_list['L2_PM2.5']=(L2_list['distance_rec_x']/L2_list['distance_rec_y'])*L2_list['PM2.5']
            df_L2 = pd.DataFrame(L2_list.groupby(['grid_ID'])['L2_PM2.5'].sum()).reset_index()

    
        #合併數據(join)
        df_all = pd.merge(pd.merge(df_center,df_L1,on='grid_ID'),df_L2,on='grid_ID')
        df_all['radius'] = radius
        df_all['interval'] = interval

        #總合併(union)
        df_goal = pd.concat([df_goal,df_all],sort=True)
    return df_goal


# In[68]:


#-----------預測第1小時地圖-----------
df1['PM2.5']=prepm25['1hr']
df1

#taichung = gp.read_file("/home/hpc/taichungcity.geojson")           #台中邊界
taichungmap_1x1 = gp.read_file("/home/gh555657/123321/final.geojson")         #台中1*1網格
taichung_district = gp.read_file("/home/gh555657/123321/taichung_district.geojson")
#list1= [   1,    4,   14,   26,   44,   63,   82,  102,  122,  144,
#         168,  193,  221,  257,  304,  353,  403,  455,  510,  568,
#         627,  687,  750,  819,  892,  968, 1053, 1141, 1232, 1325,
#        1418, 1510, 1601, 1692, 1781, 1864, 1944, 2019, 2087, 2145,
#        2197, 2246, 2289, 2330, 2359, 2384, 2403, 2419, 2433, 2445 ]
#list2= [   3,   13,   25,   43,   62,   81,  101,  121,  143,  167,
#         192,  220,  256,  303,  352,  402,  454,  509,  567,  626,
#         686,  749,  818,  891,  967, 1052, 1140, 1231, 1324, 1417,
#        1509, 1600, 1691, 1780, 1863, 1943, 2018, 2086, 2144, 2196,
#        2245, 2288, 2329, 2358, 2383, 2402, 2418, 2432, 2444, 2449 ]
lon_max=taichungmap_1x1.bounds.maxx
lon_min=taichungmap_1x1.bounds.minx
lat_max=taichungmap_1x1.bounds.maxy
lat_min=taichungmap_1x1.bounds.miny

# idw=====================================================
# lat_max,lat_min,lon_max,lon_min四份合併做成DataFrame
df_grid = pd.DataFrame([lat_max, lat_min, lon_max, lon_min]).T
df_grid['grid_Longitude'] = (df_grid['maxx'] + df_grid['minx']) / 2
df_grid['grid_Latitude'] = (df_grid['maxy'] + df_grid['miny']) / 2
df_grid['grid_ID'] = df_grid.index + 1

# 定義IDW

# 執行idw()並deepcopy切割
df_goal = idw(df1, df_grid, 5, 10)

df_goal.reset_index(inplace=True)

# 製作狀況D的圓心推估值(center_adj)
df_goal['dis_weight'] = ((df_goal['interval'] * 1.5 + df_goal['radius']) /
                         (df_goal['interval'] * 0.5 + df_goal['radius'])) - 1
df_goal['adj'] = (df_goal['L2_PM2.5'] - df_goal['L1_PM2.5'])
df_goal['center_adj'] = df_goal['L1_PM2.5'] -     df_goal['adj']*df_goal['dis_weight']
'''
依照狀況A~D，給上不同的最終推估值(est_PM2.5)
共有以下A~D四種狀況：
A.半徑5km內至少有1個觀測站，則估計值直接使用center_PM2.5。
B.半徑5km內沒有任何觀測站、第一層圓環至少有1觀測站、第二層圓環沒有任何觀測站，則估計值直接使用L1_PM2.5。
C.半徑5km內沒有任何觀測站、第一層圓環沒有任何觀測站、第二層圓環至少有1觀測站，則估計值直接使用L2_PM2.5或無法估計。
D.半徑5km內沒有任何觀測站、第一層圓環至少有1觀測站、第二層圓環至少有1觀測站，則使用L1_PM2.5和L2_PM2.5的遞減遞增估計。
E.第二圓環以內完全沒有任何觀測站，即半徑25km以內完全沒有觀測站，則不應該估計數值：無法估計。
'''
condition_A = ~df_goal['center_PM2.5'].isnull()
condition_B = (df_goal['center_PM2.5'].isnull()) & (
    ~df_goal['L1_PM2.5'].isnull()) & (df_goal['L2_PM2.5'].isnull())
condition_C = (df_goal['center_PM2.5'].isnull()) & (
    df_goal['L1_PM2.5'].isnull()) & (~df_goal['L2_PM2.5'].isnull())
condition_D = (df_goal['center_PM2.5'].isnull()) & (
    ~df_goal['L1_PM2.5'].isnull()) & (~df_goal['L2_PM2.5'].isnull())

df_goal['est_PM2.5'] = df_goal['center_PM2.5']
df_goal['est_PM2.5'].loc[condition_A] = df_goal['center_PM2.5'].loc[
    condition_A]
df_goal['est_PM2.5'].loc[condition_B] = df_goal['L1_PM2.5'].loc[condition_B]
df_goal['est_PM2.5'].loc[condition_C]=-1
#df_goal['est_PM2.5'].loc[condition_C] = df_goal['L2_PM2.5'].loc[condition_C]
df_goal['est_PM2.5'].loc[condition_D] = df_goal['center_adj'].loc[condition_D]

# 產出df3以供後續應用
df3 = pd.merge(df_goal, df_grid, how='inner', on='grid_ID')
df3 = df3[['grid_Latitude', 'grid_Longitude', 'est_PM2.5', 'grid_ID']]
df3.columns = ['Latitude', 'Longitude', 'PM2.5', 'Id']
df3.loc[df3['PM2.5'].isnull(), 'PM2.5'] = -1
# =========================================================
df3.to_csv("/home/gh555657/123321/all_point_data_epa.csv")
all_point_data_epa = pd.read_csv("/home/gh555657/123321/all_point_data_epa.csv")
lon=list(all_point_data_epa['Longitude'])
lat=list(all_point_data_epa['Latitude'])
#all_point_data_epa['Id']=0
Id=taichungmap_1x1['Id']
ans_Id=all_point_data_epa['Id']


def generalID(lon,lat,column_num,row_num):
    # 若在范围外的点，返回-1
    if lon <= LON1 or lon >= LON2 or lat <= LAT1 or lat >= LAT2:
        return -1
    # 把经度范围根据列数等分切割
    column = (LON2 - LON1)/column_num
    # 把纬度范围根据行数数等分切割
    row = (LAT2 - LAT1)/row_num
    # 二维矩阵坐标索引转换为一维ID，即： （列坐标区域（向下取整）+ 1） + （行坐标区域 * 列数）
    return int((lon-LON1)/column)+ 1 + int((lat-LAT1)/row) * column_num


taichungmap_1x1 = taichungmap_1x1.merge(all_point_data_epa, on='Id')
taichungmap_1x1['PM2.5']=taichungmap_1x1['PM2.5'].round()



df1['SiteName']=df11
df1=df1[['Id','SiteName','PM2.5','Latitude','Longitude']]


# =============================================================================================================
# folium
variable = 'PM2.5'
colorList = [(215, 207, 207, 1), '#98fb98', '#51ff51', '#00ff00', '#1ce11c',
             '#32cd32', '#ffff00', '#ffee00', '#ffd13f', '#ffc700', '#ffbf4a',
             '#ffa500', '#ff6347', '#ff5047', '#ff4c2c', '#ff0000', '#d32c4a',
             '#ba55d3']
map_color = cm.StepColormap(colorList,index=[-1,0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    vmin=-1,vmax=100,caption = 'PM2.5')

fmap = folium.Map(location=[24.2, 120.9], zoom_start=10.5)

folium.GeoJson(taichungmap_1x1,
               name='PM2.5',
               style_function=lambda x: {
                   'fillColor': map_color(x['properties'][variable]),
                   'color': 'black',
                   'weight': 0,
                   'fillOpacity': 0.7
               },
               highlight_function=lambda x: {
                   'weight': 3,
                   'color': 'black'
               },
               tooltip=folium.GeoJsonTooltip(fields=['Id', 'PM2.5'],
                                             aliases=['Id', 'PM2.5'],
                                             labels=True,
                                             sticky=True)).add_to(fmap)


#微型感測器logo
epa_micro_url= 'https://ci.taiwan.gov.tw/dsp/img/map_icon/air_quality.png'
# 環保署 logo
epa_icon_url = 'https://www.epa.gov.tw/public/MMO/epa/Epa_Logo_01_LOGO.gif'


station = folium.FeatureGroup(name="環保署", show=True)
for i in (range(15)):
    station.add_child(
        folium.Marker(
            location=[df1['Latitude'][i], df1['Longitude'][i]],
            popup=("<b>NAME:</b> {NAME}<br>"
                   " <p><b>PM2.5:</b> {PM25}<br>"
                   " <p><b>TIME:</b> {TIME}<br>").format(
                       NAME=str(df1['SiteName'][i]),
                       PM25=str(df1['PM2.5'][i]),
                       TIME=str(pr1_time)),
            icon=folium.CustomIcon(epa_icon_url,
                                   icon_size=(23,
                                              23))  # Creating a custom Icon
        ))



fmap.add_child(station)


fmap.add_child(map_color)
folium.LayerControl().add_to(fmap)
# lat/lon to map
# folium.LatLngPopup().add_to(fmap)
fmap.save('/var/www/html/predict1.html')  # 存成 final.html


# In[69]:


#-----------預測第2小時地圖-----------
df1['PM2.5']=prepm25['2hr']

#taichung = gp.read_file("/home/hpc/taichungcity.geojson")           #台中邊界
taichungmap_1x1 = gp.read_file("/home/gh555657/123321/final.geojson")         #台中1*1網格
taichung_district = gp.read_file("/home/gh555657/123321/taichung_district.geojson")
#list1= [   1,    4,   14,   26,   44,   63,   82,  102,  122,  144,
#         168,  193,  221,  257,  304,  353,  403,  455,  510,  568,
#         627,  687,  750,  819,  892,  968, 1053, 1141, 1232, 1325,
#        1418, 1510, 1601, 1692, 1781, 1864, 1944, 2019, 2087, 2145,
#        2197, 2246, 2289, 2330, 2359, 2384, 2403, 2419, 2433, 2445 ]
#list2= [   3,   13,   25,   43,   62,   81,  101,  121,  143,  167,
#         192,  220,  256,  303,  352,  402,  454,  509,  567,  626,
#         686,  749,  818,  891,  967, 1052, 1140, 1231, 1324, 1417,
#        1509, 1600, 1691, 1780, 1863, 1943, 2018, 2086, 2144, 2196,
#        2245, 2288, 2329, 2358, 2383, 2402, 2418, 2432, 2444, 2449 ]
lon_max=taichungmap_1x1.bounds.maxx
lon_min=taichungmap_1x1.bounds.minx
lat_max=taichungmap_1x1.bounds.maxy
lat_min=taichungmap_1x1.bounds.miny


# idw=====================================================
# lat_max,lat_min,lon_max,lon_min四份合併做成DataFrame
df_grid = pd.DataFrame([lat_max, lat_min, lon_max, lon_min]).T
df_grid['grid_Longitude'] = (df_grid['maxx'] + df_grid['minx']) / 2
df_grid['grid_Latitude'] = (df_grid['maxy'] + df_grid['miny']) / 2
df_grid['grid_ID'] = df_grid.index + 1

# 定義IDW

# 執行idw()並deepcopy切割
df_goal = idw(df1, df_grid, 5, 10)

df_goal.reset_index(inplace=True)

# 製作狀況D的圓心推估值(center_adj)
df_goal['dis_weight'] = ((df_goal['interval'] * 1.5 + df_goal['radius']) /
                         (df_goal['interval'] * 0.5 + df_goal['radius'])) - 1
df_goal['adj'] = (df_goal['L2_PM2.5'] - df_goal['L1_PM2.5'])
df_goal['center_adj'] = df_goal['L1_PM2.5'] -     df_goal['adj']*df_goal['dis_weight']
'''
依照狀況A~D，給上不同的最終推估值(est_PM2.5)
共有以下A~D四種狀況：
A.半徑5km內至少有1個觀測站，則估計值直接使用center_PM2.5。
B.半徑5km內沒有任何觀測站、第一層圓環至少有1觀測站、第二層圓環沒有任何觀測站，則估計值直接使用L1_PM2.5。
C.半徑5km內沒有任何觀測站、第一層圓環沒有任何觀測站、第二層圓環至少有1觀測站，則估計值直接使用L2_PM2.5或無法估計。
D.半徑5km內沒有任何觀測站、第一層圓環至少有1觀測站、第二層圓環至少有1觀測站，則使用L1_PM2.5和L2_PM2.5的遞減遞增估計。
E.第二圓環以內完全沒有任何觀測站，即半徑25km以內完全沒有觀測站，則不應該估計數值：無法估計。
'''
condition_A = ~df_goal['center_PM2.5'].isnull()
condition_B = (df_goal['center_PM2.5'].isnull()) & (
    ~df_goal['L1_PM2.5'].isnull()) & (df_goal['L2_PM2.5'].isnull())
condition_C = (df_goal['center_PM2.5'].isnull()) & (
    df_goal['L1_PM2.5'].isnull()) & (~df_goal['L2_PM2.5'].isnull())
condition_D = (df_goal['center_PM2.5'].isnull()) & (
    ~df_goal['L1_PM2.5'].isnull()) & (~df_goal['L2_PM2.5'].isnull())

df_goal['est_PM2.5'] = df_goal['center_PM2.5']
df_goal['est_PM2.5'].loc[condition_A] = df_goal['center_PM2.5'].loc[
    condition_A]
df_goal['est_PM2.5'].loc[condition_B] = df_goal['L1_PM2.5'].loc[condition_B]
df_goal['est_PM2.5'].loc[condition_C]=-1
#df_goal['est_PM2.5'].loc[condition_C] = df_goal['L2_PM2.5'].loc[condition_C]
df_goal['est_PM2.5'].loc[condition_D] = df_goal['center_adj'].loc[condition_D]

# 產出df3以供後續應用
df3 = pd.merge(df_goal, df_grid, how='inner', on='grid_ID')
df3 = df3[['grid_Latitude', 'grid_Longitude', 'est_PM2.5', 'grid_ID']]
df3.columns = ['Latitude', 'Longitude', 'PM2.5', 'Id']
df3.loc[df3['PM2.5'].isnull(), 'PM2.5'] = -1
# =========================================================
df3.to_csv("/home/gh555657/123321/all_point_data_epa.csv")
all_point_data_epa = pd.read_csv("/home/gh555657/123321/all_point_data_epa.csv")
lon=list(all_point_data_epa['Longitude'])
lat=list(all_point_data_epa['Latitude'])
#all_point_data_epa['Id']=0
Id=taichungmap_1x1['Id']
ans_Id=all_point_data_epa['Id']


def generalID(lon,lat,column_num,row_num):
    # 若在范围外的点，返回-1
    if lon <= LON1 or lon >= LON2 or lat <= LAT1 or lat >= LAT2:
        return -1
    # 把经度范围根据列数等分切割
    column = (LON2 - LON1)/column_num
    # 把纬度范围根据行数数等分切割
    row = (LAT2 - LAT1)/row_num
    # 二维矩阵坐标索引转换为一维ID，即： （列坐标区域（向下取整）+ 1） + （行坐标区域 * 列数）
    return int((lon-LON1)/column)+ 1 + int((lat-LAT1)/row) * column_num


taichungmap_1x1 = taichungmap_1x1.merge(all_point_data_epa, on='Id')
taichungmap_1x1['PM2.5']=taichungmap_1x1['PM2.5'].round()



df1['SiteName']=df11
df1=df1[['Id','SiteName','PM2.5','Latitude','Longitude']]


# =============================================================================================================
# folium

variable = 'PM2.5'
colorList = [(215, 207, 207, 1), '#98fb98', '#51ff51', '#00ff00', '#1ce11c',
             '#32cd32', '#ffff00', '#ffee00', '#ffd13f', '#ffc700', '#ffbf4a',
             '#ffa500', '#ff6347', '#ff5047', '#ff4c2c', '#ff0000', '#d32c4a',
             '#ba55d3']
map_color = cm.StepColormap(colorList,index=[-1,0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    vmin=-1,vmax=100,caption = 'PM2.5')

fmap = folium.Map(location=[24.2, 120.9], zoom_start=10.5)

folium.GeoJson(taichungmap_1x1,
               name='PM2.5',
               style_function=lambda x: {
                   'fillColor': map_color(x['properties'][variable]),
                   'color': 'black',
                   'weight': 0,
                   'fillOpacity': 0.7
               },
               highlight_function=lambda x: {
                   'weight': 3,
                   'color': 'black'
               },
               tooltip=folium.GeoJsonTooltip(fields=['Id', 'PM2.5'],
                                             aliases=['Id', 'PM2.5'],
                                             labels=True,
                                             sticky=True)).add_to(fmap)


#微型感測器logo
epa_micro_url= 'https://ci.taiwan.gov.tw/dsp/img/map_icon/air_quality.png'
# 環保署 logo
epa_icon_url = 'https://www.epa.gov.tw/public/MMO/epa/Epa_Logo_01_LOGO.gif'


station = folium.FeatureGroup(name="環保署", show=True)
for i in (range(15)):
    station.add_child(
        folium.Marker(
            location=[df1['Latitude'][i], df1['Longitude'][i]],
            popup=("<b>NAME:</b> {NAME}<br>"
                   " <p><b>PM2.5:</b> {PM25}<br>"
                   " <p><b>TIME:</b> {TIME}<br>").format(
                       NAME=str(df1['SiteName'][i]),
                       PM25=str(df1['PM2.5'][i]),
                       TIME=str(pr2_time)),
            icon=folium.CustomIcon(epa_icon_url,
                                   icon_size=(23,
                                              23))  # Creating a custom Icon
        ))



fmap.add_child(station)


fmap.add_child(map_color)
folium.LayerControl().add_to(fmap)
# lat/lon to map
# folium.LatLngPopup().add_to(fmap)
fmap.save('/var/www/html/predict2.html')  # 存成 final.html


# In[70]:


#-----------預測第3小時地圖-----------
df1['PM2.5']=prepm25['3hr']

#taichung = gp.read_file("/home/hpc/taichungcity.geojson")           #台中邊界
taichungmap_1x1 = gp.read_file("/home/gh555657/123321/final.geojson")         #台中1*1網格
taichung_district = gp.read_file("/home/gh555657/123321/taichung_district.geojson")
#list1= [   1,    4,   14,   26,   44,   63,   82,  102,  122,  144,
#         168,  193,  221,  257,  304,  353,  403,  455,  510,  568,
#         627,  687,  750,  819,  892,  968, 1053, 1141, 1232, 1325,
#        1418, 1510, 1601, 1692, 1781, 1864, 1944, 2019, 2087, 2145,
#        2197, 2246, 2289, 2330, 2359, 2384, 2403, 2419, 2433, 2445 ]
#list2= [   3,   13,   25,   43,   62,   81,  101,  121,  143,  167,
#         192,  220,  256,  303,  352,  402,  454,  509,  567,  626,
#         686,  749,  818,  891,  967, 1052, 1140, 1231, 1324, 1417,
#        1509, 1600, 1691, 1780, 1863, 1943, 2018, 2086, 2144, 2196,
#        2245, 2288, 2329, 2358, 2383, 2402, 2418, 2432, 2444, 2449 ]
lon_max=taichungmap_1x1.bounds.maxx
lon_min=taichungmap_1x1.bounds.minx
lat_max=taichungmap_1x1.bounds.maxy
lat_min=taichungmap_1x1.bounds.miny



# idw=====================================================
# lat_max,lat_min,lon_max,lon_min四份合併做成DataFrame
df_grid = pd.DataFrame([lat_max, lat_min, lon_max, lon_min]).T
df_grid['grid_Longitude'] = (df_grid['maxx'] + df_grid['minx']) / 2
df_grid['grid_Latitude'] = (df_grid['maxy'] + df_grid['miny']) / 2
df_grid['grid_ID'] = df_grid.index + 1

# 定義IDW

# 執行idw()並deepcopy切割
df_goal = idw(df1, df_grid, 5, 10)

df_goal.reset_index(inplace=True)

# 製作狀況D的圓心推估值(center_adj)
df_goal['dis_weight'] = ((df_goal['interval'] * 1.5 + df_goal['radius']) /
                         (df_goal['interval'] * 0.5 + df_goal['radius'])) - 1
df_goal['adj'] = (df_goal['L2_PM2.5'] - df_goal['L1_PM2.5'])
df_goal['center_adj'] = df_goal['L1_PM2.5'] -     df_goal['adj']*df_goal['dis_weight']
'''
依照狀況A~D，給上不同的最終推估值(est_PM2.5)
共有以下A~D四種狀況：
A.半徑5km內至少有1個觀測站，則估計值直接使用center_PM2.5。
B.半徑5km內沒有任何觀測站、第一層圓環至少有1觀測站、第二層圓環沒有任何觀測站，則估計值直接使用L1_PM2.5。
C.半徑5km內沒有任何觀測站、第一層圓環沒有任何觀測站、第二層圓環至少有1觀測站，則估計值直接使用L2_PM2.5或無法估計。
D.半徑5km內沒有任何觀測站、第一層圓環至少有1觀測站、第二層圓環至少有1觀測站，則使用L1_PM2.5和L2_PM2.5的遞減遞增估計。
E.第二圓環以內完全沒有任何觀測站，即半徑25km以內完全沒有觀測站，則不應該估計數值：無法估計。
'''
condition_A = ~df_goal['center_PM2.5'].isnull()
condition_B = (df_goal['center_PM2.5'].isnull()) & (
    ~df_goal['L1_PM2.5'].isnull()) & (df_goal['L2_PM2.5'].isnull())
condition_C = (df_goal['center_PM2.5'].isnull()) & (
    df_goal['L1_PM2.5'].isnull()) & (~df_goal['L2_PM2.5'].isnull())
condition_D = (df_goal['center_PM2.5'].isnull()) & (
    ~df_goal['L1_PM2.5'].isnull()) & (~df_goal['L2_PM2.5'].isnull())

df_goal['est_PM2.5'] = df_goal['center_PM2.5']
df_goal['est_PM2.5'].loc[condition_A] = df_goal['center_PM2.5'].loc[
    condition_A]
df_goal['est_PM2.5'].loc[condition_B] = df_goal['L1_PM2.5'].loc[condition_B]
df_goal['est_PM2.5'].loc[condition_C]=-1
#df_goal['est_PM2.5'].loc[condition_C] = df_goal['L2_PM2.5'].loc[condition_C]
df_goal['est_PM2.5'].loc[condition_D] = df_goal['center_adj'].loc[condition_D]

# 產出df3以供後續應用
df3 = pd.merge(df_goal, df_grid, how='inner', on='grid_ID')
df3 = df3[['grid_Latitude', 'grid_Longitude', 'est_PM2.5', 'grid_ID']]
df3.columns = ['Latitude', 'Longitude', 'PM2.5', 'Id']
df3.loc[df3['PM2.5'].isnull(), 'PM2.5'] = -1
# =========================================================
df3.to_csv("/home/gh555657/123321/all_point_data_epa.csv")
all_point_data_epa = pd.read_csv("/home/gh555657/123321/all_point_data_epa.csv")
lon=list(all_point_data_epa['Longitude'])
lat=list(all_point_data_epa['Latitude'])
#all_point_data_epa['Id']=0
Id=taichungmap_1x1['Id']
ans_Id=all_point_data_epa['Id']


def generalID(lon,lat,column_num,row_num):
    # 若在范围外的点，返回-1
    if lon <= LON1 or lon >= LON2 or lat <= LAT1 or lat >= LAT2:
        return -1
    # 把经度范围根据列数等分切割
    column = (LON2 - LON1)/column_num
    # 把纬度范围根据行数数等分切割
    row = (LAT2 - LAT1)/row_num
    # 二维矩阵坐标索引转换为一维ID，即： （列坐标区域（向下取整）+ 1） + （行坐标区域 * 列数）
    return int((lon-LON1)/column)+ 1 + int((lat-LAT1)/row) * column_num


taichungmap_1x1 = taichungmap_1x1.merge(all_point_data_epa, on='Id')
taichungmap_1x1['PM2.5']=taichungmap_1x1['PM2.5'].round()



df1['SiteName']=df11
df1=df1[['Id','SiteName','PM2.5','Latitude','Longitude']]


# =============================================================================================================
# folium
variable = 'PM2.5'
colorList = [(215, 207, 207, 1), '#98fb98', '#51ff51', '#00ff00', '#1ce11c',
             '#32cd32', '#ffff00', '#ffee00', '#ffd13f', '#ffc700', '#ffbf4a',
             '#ffa500', '#ff6347', '#ff5047', '#ff4c2c', '#ff0000', '#d32c4a',
             '#ba55d3']
map_color = cm.StepColormap(colorList,index=[-1,0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    vmin=-1,vmax=100,caption = 'PM2.5')

fmap = folium.Map(location=[24.2, 120.9], zoom_start=10.5)

folium.GeoJson(taichungmap_1x1,
               name='PM2.5',
               style_function=lambda x: {
                   'fillColor': map_color(x['properties'][variable]),
                   'color': 'black',
                   'weight': 0,
                   'fillOpacity': 0.7
               },
               highlight_function=lambda x: {
                   'weight': 3,
                   'color': 'black'
               },
               tooltip=folium.GeoJsonTooltip(fields=['Id', 'PM2.5'],
                                             aliases=['Id', 'PM2.5'],
                                             labels=True,
                                             sticky=True)).add_to(fmap)


#微型感測器logo
epa_micro_url= 'https://ci.taiwan.gov.tw/dsp/img/map_icon/air_quality.png'
# 環保署 logo
epa_icon_url = 'https://www.epa.gov.tw/public/MMO/epa/Epa_Logo_01_LOGO.gif'


station = folium.FeatureGroup(name="環保署", show=True)
for i in (range(15)):
    station.add_child(
        folium.Marker(
            location=[df1['Latitude'][i], df1['Longitude'][i]],
            popup=("<b>NAME:</b> {NAME}<br>"
                   " <p><b>PM2.5:</b> {PM25}<br>"
                   " <p><b>TIME:</b> {TIME}<br>").format(
                       NAME=str(df1['SiteName'][i]),
                       PM25=str(df1['PM2.5'][i]),
                       TIME=str(pr3_time)),
            icon=folium.CustomIcon(epa_icon_url,
                                   icon_size=(23,
                                              23))  # Creating a custom Icon
        ))



fmap.add_child(station)


fmap.add_child(map_color)
folium.LayerControl().add_to(fmap)
# lat/lon to map
# folium.LatLngPopup().add_to(fmap)
fmap.save('/var/www/html/predict3.html')  # 存成 final.html


# In[71]:


#-----------預測第4小時地圖-----------
df1['PM2.5']=prepm25['4hr']


#taichung = gp.read_file("/home/hpc/taichungcity.geojson")           #台中邊界
taichungmap_1x1 = gp.read_file("/home/gh555657/123321/final.geojson")         #台中1*1網格
taichung_district = gp.read_file("/home/gh555657/123321/taichung_district.geojson")
#list1= [   1,    4,   14,   26,   44,   63,   82,  102,  122,  144,
#         168,  193,  221,  257,  304,  353,  403,  455,  510,  568,
#         627,  687,  750,  819,  892,  968, 1053, 1141, 1232, 1325,
#        1418, 1510, 1601, 1692, 1781, 1864, 1944, 2019, 2087, 2145,
#        2197, 2246, 2289, 2330, 2359, 2384, 2403, 2419, 2433, 2445 ]
#list2= [   3,   13,   25,   43,   62,   81,  101,  121,  143,  167,
#         192,  220,  256,  303,  352,  402,  454,  509,  567,  626,
#         686,  749,  818,  891,  967, 1052, 1140, 1231, 1324, 1417,
#        1509, 1600, 1691, 1780, 1863, 1943, 2018, 2086, 2144, 2196,
#        2245, 2288, 2329, 2358, 2383, 2402, 2418, 2432, 2444, 2449 ]
lon_max=taichungmap_1x1.bounds.maxx
lon_min=taichungmap_1x1.bounds.minx
lat_max=taichungmap_1x1.bounds.maxy
lat_min=taichungmap_1x1.bounds.miny



# idw=====================================================
# lat_max,lat_min,lon_max,lon_min四份合併做成DataFrame
df_grid = pd.DataFrame([lat_max, lat_min, lon_max, lon_min]).T
df_grid['grid_Longitude'] = (df_grid['maxx'] + df_grid['minx']) / 2
df_grid['grid_Latitude'] = (df_grid['maxy'] + df_grid['miny']) / 2
df_grid['grid_ID'] = df_grid.index + 1

# 定義IDW

# 執行idw()並deepcopy切割
df_goal = idw(df1, df_grid, 5, 10)

df_goal.reset_index(inplace=True)

# 製作狀況D的圓心推估值(center_adj)
df_goal['dis_weight'] = ((df_goal['interval'] * 1.5 + df_goal['radius']) /
                         (df_goal['interval'] * 0.5 + df_goal['radius'])) - 1
df_goal['adj'] = (df_goal['L2_PM2.5'] - df_goal['L1_PM2.5'])
df_goal['center_adj'] = df_goal['L1_PM2.5'] -     df_goal['adj']*df_goal['dis_weight']
'''
依照狀況A~D，給上不同的最終推估值(est_PM2.5)
共有以下A~D四種狀況：
A.半徑5km內至少有1個觀測站，則估計值直接使用center_PM2.5。
B.半徑5km內沒有任何觀測站、第一層圓環至少有1觀測站、第二層圓環沒有任何觀測站，則估計值直接使用L1_PM2.5。
C.半徑5km內沒有任何觀測站、第一層圓環沒有任何觀測站、第二層圓環至少有1觀測站，則估計值直接使用L2_PM2.5或無法估計。
D.半徑5km內沒有任何觀測站、第一層圓環至少有1觀測站、第二層圓環至少有1觀測站，則使用L1_PM2.5和L2_PM2.5的遞減遞增估計。
E.第二圓環以內完全沒有任何觀測站，即半徑25km以內完全沒有觀測站，則不應該估計數值：無法估計。
'''
condition_A = ~df_goal['center_PM2.5'].isnull()
condition_B = (df_goal['center_PM2.5'].isnull()) & (
    ~df_goal['L1_PM2.5'].isnull()) & (df_goal['L2_PM2.5'].isnull())
condition_C = (df_goal['center_PM2.5'].isnull()) & (
    df_goal['L1_PM2.5'].isnull()) & (~df_goal['L2_PM2.5'].isnull())
condition_D = (df_goal['center_PM2.5'].isnull()) & (
    ~df_goal['L1_PM2.5'].isnull()) & (~df_goal['L2_PM2.5'].isnull())

df_goal['est_PM2.5'] = df_goal['center_PM2.5']
df_goal['est_PM2.5'].loc[condition_A] = df_goal['center_PM2.5'].loc[
    condition_A]
df_goal['est_PM2.5'].loc[condition_B] = df_goal['L1_PM2.5'].loc[condition_B]
df_goal['est_PM2.5'].loc[condition_C]=-1
#df_goal['est_PM2.5'].loc[condition_C] = df_goal['L2_PM2.5'].loc[condition_C]
df_goal['est_PM2.5'].loc[condition_D] = df_goal['center_adj'].loc[condition_D]

# 產出df3以供後續應用
df3 = pd.merge(df_goal, df_grid, how='inner', on='grid_ID')
df3 = df3[['grid_Latitude', 'grid_Longitude', 'est_PM2.5', 'grid_ID']]
df3.columns = ['Latitude', 'Longitude', 'PM2.5', 'Id']
df3.loc[df3['PM2.5'].isnull(), 'PM2.5'] = -1
# =========================================================
df3.to_csv("/home/gh555657/123321/all_point_data_epa.csv")
all_point_data_epa = pd.read_csv("/home/gh555657/123321/all_point_data_epa.csv")
lon=list(all_point_data_epa['Longitude'])
lat=list(all_point_data_epa['Latitude'])
#all_point_data_epa['Id']=0
Id=taichungmap_1x1['Id']
ans_Id=all_point_data_epa['Id']


def generalID(lon,lat,column_num,row_num):
    # 若在范围外的点，返回-1
    if lon <= LON1 or lon >= LON2 or lat <= LAT1 or lat >= LAT2:
        return -1
    # 把经度范围根据列数等分切割
    column = (LON2 - LON1)/column_num
    # 把纬度范围根据行数数等分切割
    row = (LAT2 - LAT1)/row_num
    # 二维矩阵坐标索引转换为一维ID，即： （列坐标区域（向下取整）+ 1） + （行坐标区域 * 列数）
    return int((lon-LON1)/column)+ 1 + int((lat-LAT1)/row) * column_num


taichungmap_1x1 = taichungmap_1x1.merge(all_point_data_epa, on='Id')
taichungmap_1x1['PM2.5']=taichungmap_1x1['PM2.5'].round()



df1['SiteName']=df11
df1=df1[['Id','SiteName','PM2.5','Latitude','Longitude']]


# =============================================================================================================
# folium
variable = 'PM2.5'
colorList = [(215, 207, 207, 1), '#98fb98', '#51ff51', '#00ff00', '#1ce11c',
             '#32cd32', '#ffff00', '#ffee00', '#ffd13f', '#ffc700', '#ffbf4a',
             '#ffa500', '#ff6347', '#ff5047', '#ff4c2c', '#ff0000', '#d32c4a',
             '#ba55d3']
map_color = cm.StepColormap(colorList,index=[-1,0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    vmin=-1,vmax=100,caption = 'PM2.5')

fmap = folium.Map(location=[24.2, 120.9], zoom_start=10.5)

folium.GeoJson(taichungmap_1x1,
               name='PM2.5',
               style_function=lambda x: {
                   'fillColor': map_color(x['properties'][variable]),
                   'color': 'black',
                   'weight': 0,
                   'fillOpacity': 0.7
               },
               highlight_function=lambda x: {
                   'weight': 3,
                   'color': 'black'
               },
               tooltip=folium.GeoJsonTooltip(fields=['Id', 'PM2.5'],
                                             aliases=['Id', 'PM2.5'],
                                             labels=True,
                                             sticky=True)).add_to(fmap)


#微型感測器logo
epa_micro_url= 'https://ci.taiwan.gov.tw/dsp/img/map_icon/air_quality.png'
# 環保署 logo
epa_icon_url = 'https://www.epa.gov.tw/public/MMO/epa/Epa_Logo_01_LOGO.gif'


station = folium.FeatureGroup(name="環保署", show=True)
for i in (range(15)):
    station.add_child(
        folium.Marker(
            location=[df1['Latitude'][i], df1['Longitude'][i]],
            popup=("<b>NAME:</b> {NAME}<br>"
                   " <p><b>PM2.5:</b> {PM25}<br>"
                   " <p><b>TIME:</b> {TIME}<br>").format(
                       NAME=str(df1['SiteName'][i]),
                       PM25=str(df1['PM2.5'][i]),
                       TIME=str(pr4_time)),
            icon=folium.CustomIcon(epa_icon_url,
                                   icon_size=(23,
                                              23))  # Creating a custom Icon
        ))



fmap.add_child(station)


fmap.add_child(map_color)
folium.LayerControl().add_to(fmap)
# lat/lon to map
# folium.LatLngPopup().add_to(fmap)
fmap.save('/var/www/html/predict4.html')  # 存成 final.html


# In[72]:


#-----------預測第5小時地圖-----------
df1['PM2.5']=prepm25['5hr']


#taichung = gp.read_file("/home/hpc/taichungcity.geojson")           #台中邊界
taichungmap_1x1 = gp.read_file("/home/gh555657/123321/final.geojson")         #台中1*1網格
taichung_district = gp.read_file("/home/gh555657/123321/taichung_district.geojson")
#list1= [   1,    4,   14,   26,   44,   63,   82,  102,  122,  144,
#         168,  193,  221,  257,  304,  353,  403,  455,  510,  568,
#         627,  687,  750,  819,  892,  968, 1053, 1141, 1232, 1325,
#        1418, 1510, 1601, 1692, 1781, 1864, 1944, 2019, 2087, 2145,
#        2197, 2246, 2289, 2330, 2359, 2384, 2403, 2419, 2433, 2445 ]
#list2= [   3,   13,   25,   43,   62,   81,  101,  121,  143,  167,
#         192,  220,  256,  303,  352,  402,  454,  509,  567,  626,
#         686,  749,  818,  891,  967, 1052, 1140, 1231, 1324, 1417,
#        1509, 1600, 1691, 1780, 1863, 1943, 2018, 2086, 2144, 2196,
#        2245, 2288, 2329, 2358, 2383, 2402, 2418, 2432, 2444, 2449 ]
lon_max=taichungmap_1x1.bounds.maxx
lon_min=taichungmap_1x1.bounds.minx
lat_max=taichungmap_1x1.bounds.maxy
lat_min=taichungmap_1x1.bounds.miny


# idw=====================================================
# lat_max,lat_min,lon_max,lon_min四份合併做成DataFrame
df_grid = pd.DataFrame([lat_max, lat_min, lon_max, lon_min]).T
df_grid['grid_Longitude'] = (df_grid['maxx'] + df_grid['minx']) / 2
df_grid['grid_Latitude'] = (df_grid['maxy'] + df_grid['miny']) / 2
df_grid['grid_ID'] = df_grid.index + 1

# 定義IDW

# 執行idw()並deepcopy切割
df_goal = idw(df1, df_grid, 5, 10)

df_goal.reset_index(inplace=True)

# 製作狀況D的圓心推估值(center_adj)
df_goal['dis_weight'] = ((df_goal['interval'] * 1.5 + df_goal['radius']) /
                         (df_goal['interval'] * 0.5 + df_goal['radius'])) - 1
df_goal['adj'] = (df_goal['L2_PM2.5'] - df_goal['L1_PM2.5'])
df_goal['center_adj'] = df_goal['L1_PM2.5'] -     df_goal['adj']*df_goal['dis_weight']
'''
依照狀況A~D，給上不同的最終推估值(est_PM2.5)
共有以下A~D四種狀況：
A.半徑5km內至少有1個觀測站，則估計值直接使用center_PM2.5。
B.半徑5km內沒有任何觀測站、第一層圓環至少有1觀測站、第二層圓環沒有任何觀測站，則估計值直接使用L1_PM2.5。
C.半徑5km內沒有任何觀測站、第一層圓環沒有任何觀測站、第二層圓環至少有1觀測站，則估計值直接使用L2_PM2.5或無法估計。
D.半徑5km內沒有任何觀測站、第一層圓環至少有1觀測站、第二層圓環至少有1觀測站，則使用L1_PM2.5和L2_PM2.5的遞減遞增估計。
E.第二圓環以內完全沒有任何觀測站，即半徑25km以內完全沒有觀測站，則不應該估計數值：無法估計。
'''
condition_A = ~df_goal['center_PM2.5'].isnull()
condition_B = (df_goal['center_PM2.5'].isnull()) & (
    ~df_goal['L1_PM2.5'].isnull()) & (df_goal['L2_PM2.5'].isnull())
condition_C = (df_goal['center_PM2.5'].isnull()) & (
    df_goal['L1_PM2.5'].isnull()) & (~df_goal['L2_PM2.5'].isnull())
condition_D = (df_goal['center_PM2.5'].isnull()) & (
    ~df_goal['L1_PM2.5'].isnull()) & (~df_goal['L2_PM2.5'].isnull())

df_goal['est_PM2.5'] = df_goal['center_PM2.5']
df_goal['est_PM2.5'].loc[condition_A] = df_goal['center_PM2.5'].loc[
    condition_A]
df_goal['est_PM2.5'].loc[condition_B] = df_goal['L1_PM2.5'].loc[condition_B]
df_goal['est_PM2.5'].loc[condition_C]=-1
#df_goal['est_PM2.5'].loc[condition_C] = df_goal['L2_PM2.5'].loc[condition_C]
df_goal['est_PM2.5'].loc[condition_D] = df_goal['center_adj'].loc[condition_D]

# 產出df3以供後續應用
df3 = pd.merge(df_goal, df_grid, how='inner', on='grid_ID')
df3 = df3[['grid_Latitude', 'grid_Longitude', 'est_PM2.5', 'grid_ID']]
df3.columns = ['Latitude', 'Longitude', 'PM2.5', 'Id']
df3.loc[df3['PM2.5'].isnull(), 'PM2.5'] = -1
# =========================================================
df3.to_csv("/home/gh555657/123321/all_point_data_epa.csv")
all_point_data_epa = pd.read_csv("/home/gh555657/123321/all_point_data_epa.csv")
lon=list(all_point_data_epa['Longitude'])
lat=list(all_point_data_epa['Latitude'])
#all_point_data_epa['Id']=0
Id=taichungmap_1x1['Id']
ans_Id=all_point_data_epa['Id']


def generalID(lon,lat,column_num,row_num):
    # 若在范围外的点，返回-1
    if lon <= LON1 or lon >= LON2 or lat <= LAT1 or lat >= LAT2:
        return -1
    # 把经度范围根据列数等分切割
    column = (LON2 - LON1)/column_num
    # 把纬度范围根据行数数等分切割
    row = (LAT2 - LAT1)/row_num
    # 二维矩阵坐标索引转换为一维ID，即： （列坐标区域（向下取整）+ 1） + （行坐标区域 * 列数）
    return int((lon-LON1)/column)+ 1 + int((lat-LAT1)/row) * column_num


taichungmap_1x1 = taichungmap_1x1.merge(all_point_data_epa, on='Id')
taichungmap_1x1['PM2.5']=taichungmap_1x1['PM2.5'].round()



df1['SiteName']=df11
df1=df1[['Id','SiteName','PM2.5','Latitude','Longitude']]


# =============================================================================================================
# folium
#(215, 207, 207, 0.00)
variable = 'PM2.5'
colorList = [(215, 207, 207, 1), '#98fb98', '#51ff51', '#00ff00', '#1ce11c',
             '#32cd32', '#ffff00', '#ffee00', '#ffd13f', '#ffc700', '#ffbf4a',
             '#ffa500', '#ff6347', '#ff5047', '#ff4c2c', '#ff0000', '#d32c4a',
             '#ba55d3']
map_color = cm.StepColormap(colorList,index=[-1,0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    vmin=-1,vmax=100,caption = 'PM2.5')

fmap = folium.Map(location=[24.2, 120.9], zoom_start=10.5)

folium.GeoJson(taichungmap_1x1,
               name='PM2.5',
               style_function=lambda x: {
                   'fillColor': map_color(x['properties'][variable]),
                   'color': 'black',
                   'weight': 0,
                   'fillOpacity': 0.7
               },
               highlight_function=lambda x: {
                   'weight': 3,
                   'color': 'black'
               },
               tooltip=folium.GeoJsonTooltip(fields=['Id', 'PM2.5'],
                                             aliases=['Id', 'PM2.5'],
                                             labels=True,
                                             sticky=True)).add_to(fmap)


#微型感測器logo
epa_micro_url= 'https://ci.taiwan.gov.tw/dsp/img/map_icon/air_quality.png'
# 環保署 logo
epa_icon_url = 'https://www.epa.gov.tw/public/MMO/epa/Epa_Logo_01_LOGO.gif'


station = folium.FeatureGroup(name="環保署", show=True)
for i in (range(15)):
    station.add_child(
        folium.Marker(
            location=[df1['Latitude'][i], df1['Longitude'][i]],
            popup=("<b>NAME:</b> {NAME}<br>"
                   " <p><b>PM2.5:</b> {PM25}<br>"
                   " <p><b>TIME:</b> {TIME}<br>").format(
                       NAME=str(df1['SiteName'][i]),
                       PM25=str(df1['PM2.5'][i]),
                       TIME=str(pr5_time)),
            icon=folium.CustomIcon(epa_icon_url,
                                   icon_size=(23,
                                              23))  # Creating a custom Icon
        ))



fmap.add_child(station)


fmap.add_child(map_color)
folium.LayerControl().add_to(fmap)
# lat/lon to map
# folium.LatLngPopup().add_to(fmap)
fmap.save('/var/www/html/predict5.html')  # 存成 final.html


# In[73]:


# -----------預測第6小時地圖-----------
df1['PM2.5'] = prepm25['6hr']

# taichung = gp.read_file("/home/hpc/taichungcity.geojson")           #台中邊界
taichungmap_1x1 = gp.read_file(
    "/home/gh555657/123321/final.geojson")  # 台中1*1網格
taichung_district = gp.read_file(
    "/home/gh555657/123321/taichung_district.geojson")
# list1= [   1,    4,   14,   26,   44,   63,   82,  102,  122,  144,
#         168,  193,  221,  257,  304,  353,  403,  455,  510,  568,
#         627,  687,  750,  819,  892,  968, 1053, 1141, 1232, 1325,
#        1418, 1510, 1601, 1692, 1781, 1864, 1944, 2019, 2087, 2145,
#        2197, 2246, 2289, 2330, 2359, 2384, 2403, 2419, 2433, 2445 ]
# list2= [   3,   13,   25,   43,   62,   81,  101,  121,  143,  167,
#         192,  220,  256,  303,  352,  402,  454,  509,  567,  626,
#         686,  749,  818,  891,  967, 1052, 1140, 1231, 1324, 1417,
#        1509, 1600, 1691, 1780, 1863, 1943, 2018, 2086, 2144, 2196,
#        2245, 2288, 2329, 2358, 2383, 2402, 2418, 2432, 2444, 2449 ]
lon_max = taichungmap_1x1.bounds.maxx
lon_min = taichungmap_1x1.bounds.minx
lat_max = taichungmap_1x1.bounds.maxy
lat_min = taichungmap_1x1.bounds.miny

# idw=====================================================
# lat_max,lat_min,lon_max,lon_min四份合併做成DataFrame
df_grid = pd.DataFrame([lat_max, lat_min, lon_max, lon_min]).T
df_grid['grid_Longitude'] = (df_grid['maxx'] + df_grid['minx']) / 2
df_grid['grid_Latitude'] = (df_grid['maxy'] + df_grid['miny']) / 2
df_grid['grid_ID'] = df_grid.index + 1

# 定義IDW

# 執行idw()並deepcopy切割
df_goal = idw(df1, df_grid, 5, 10)

df_goal.reset_index(inplace=True)

# 製作狀況D的圓心推估值(center_adj)
df_goal['dis_weight'] = ((df_goal['interval'] * 1.5 + df_goal['radius']) /
                         (df_goal['interval'] * 0.5 + df_goal['radius'])) - 1
df_goal['adj'] = (df_goal['L2_PM2.5'] - df_goal['L1_PM2.5'])
df_goal['center_adj'] = df_goal['L1_PM2.5'] -     df_goal['adj']*df_goal['dis_weight']
'''
依照狀況A~D，給上不同的最終推估值(est_PM2.5)
共有以下A~D四種狀況：
A.半徑5km內至少有1個觀測站，則估計值直接使用center_PM2.5。
B.半徑5km內沒有任何觀測站、第一層圓環至少有1觀測站、第二層圓環沒有任何觀測站，則估計值直接使用L1_PM2.5。
C.半徑5km內沒有任何觀測站、第一層圓環沒有任何觀測站、第二層圓環至少有1觀測站，則估計值直接使用L2_PM2.5或無法估計。
D.半徑5km內沒有任何觀測站、第一層圓環至少有1觀測站、第二層圓環至少有1觀測站，則使用L1_PM2.5和L2_PM2.5的遞減遞增估計。
E.第二圓環以內完全沒有任何觀測站，即半徑25km以內完全沒有觀測站，則不應該估計數值：無法估計。
'''
condition_A = ~df_goal['center_PM2.5'].isnull()
condition_B = (df_goal['center_PM2.5'].isnull()) & (
    ~df_goal['L1_PM2.5'].isnull()) & (df_goal['L2_PM2.5'].isnull())
condition_C = (df_goal['center_PM2.5'].isnull()) & (
    df_goal['L1_PM2.5'].isnull()) & (~df_goal['L2_PM2.5'].isnull())
condition_D = (df_goal['center_PM2.5'].isnull()) & (
    ~df_goal['L1_PM2.5'].isnull()) & (~df_goal['L2_PM2.5'].isnull())

df_goal['est_PM2.5'] = df_goal['center_PM2.5']
df_goal['est_PM2.5'].loc[condition_A] = df_goal['center_PM2.5'].loc[
    condition_A]
df_goal['est_PM2.5'].loc[condition_B] = df_goal['L1_PM2.5'].loc[condition_B]
df_goal['est_PM2.5'].loc[condition_C]=-1
#df_goal['est_PM2.5'].loc[condition_C] = df_goal['L2_PM2.5'].loc[condition_C]
df_goal['est_PM2.5'].loc[condition_D] = df_goal['center_adj'].loc[condition_D]

# 產出df3以供後續應用
df3 = pd.merge(df_goal, df_grid, how='inner', on='grid_ID')
df3 = df3[['grid_Latitude', 'grid_Longitude', 'est_PM2.5', 'grid_ID']]
df3.columns = ['Latitude', 'Longitude', 'PM2.5', 'Id']
df3.loc[df3['PM2.5'].isnull(), 'PM2.5'] = -1
# =========================================================
df3.to_csv("/home/gh555657/123321/all_point_data_epa.csv")
all_point_data_epa = pd.read_csv(
    "/home/gh555657/123321/all_point_data_epa.csv")
lon = list(all_point_data_epa['Longitude'])
lat = list(all_point_data_epa['Latitude'])
# all_point_data_epa['Id']=0
Id = taichungmap_1x1['Id']
ans_Id = all_point_data_epa['Id']


def generalID(lon, lat, column_num, row_num):
    # 若在范围外的点，返回-1
    if lon <= LON1 or lon >= LON2 or lat <= LAT1 or lat >= LAT2:
        return -1
    # 把经度范围根据列数等分切割
    column = (LON2 - LON1) / column_num
    # 把纬度范围根据行数数等分切割
    row = (LAT2 - LAT1) / row_num
    # 二维矩阵坐标索引转换为一维ID，即： （列坐标区域（向下取整）+ 1） + （行坐标区域 * 列数）
    return int((lon - LON1) / column) + 1 + int(
        (lat - LAT1) / row) * column_num


taichungmap_1x1 = taichungmap_1x1.merge(all_point_data_epa, on='Id')
taichungmap_1x1['PM2.5'] = taichungmap_1x1['PM2.5'].round()

df1['SiteName'] = df11
df1 = df1[['Id', 'SiteName', 'PM2.5', 'Latitude', 'Longitude']]

# =============================================================================================================
# folium
#(215, 207, 207, 0.00)
variable = 'PM2.5'
colorList = [(215, 207, 207, 1), '#98fb98', '#51ff51', '#00ff00', '#1ce11c',
             '#32cd32', '#ffff00', '#ffee00', '#ffd13f', '#ffc700', '#ffbf4a',
             '#ffa500', '#ff6347', '#ff5047', '#ff4c2c', '#ff0000', '#d32c4a',
             '#ba55d3']
map_color = cm.StepColormap(colorList,index=[-1,0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    vmin=-1,vmax=100,caption = 'PM2.5')

fmap = folium.Map(location=[24.2, 120.9], zoom_start=10.5)

folium.GeoJson(taichungmap_1x1,
               name='PM2.5',
               style_function=lambda x: {
                   'fillColor': map_color(x['properties'][variable]),
                   'color': 'black',
                   'weight': 0,
                   'fillOpacity': 0.7
               },
               highlight_function=lambda x: {
                   'weight': 3,
                   'color': 'black'
               },
               tooltip=folium.GeoJsonTooltip(fields=['Id', 'PM2.5'],
                                             aliases=['Id', 'PM2.5'],
                                             labels=True,
                                             sticky=True)).add_to(fmap)

# 微型感測器logo
epa_micro_url = 'https://ci.taiwan.gov.tw/dsp/img/map_icon/air_quality.png'
# 環保署 logo
epa_icon_url = 'https://www.epa.gov.tw/public/MMO/epa/Epa_Logo_01_LOGO.gif'

station = folium.FeatureGroup(name="環保署", show=True)
for i in (range(15)):
    station.add_child(
        folium.Marker(
            location=[df1['Latitude'][i], df1['Longitude'][i]],
            popup=("<b>NAME:</b> {NAME}<br>"
                   " <p><b>PM2.5:</b> {PM25}<br>"
                   " <p><b>TIME:</b> {TIME}<br>").format(
                       NAME=str(df1['SiteName'][i]),
                       PM25=str(df1['PM2.5'][i]),
                       TIME=str(pr6_time)),
            icon=folium.CustomIcon(epa_icon_url,
                                   icon_size=(23,
                                              23))  # Creating a custom Icon
        ))

fmap.add_child(station)

fmap.add_child(map_color)
folium.LayerControl().add_to(fmap)
# lat/lon to map
# folium.LatLngPopup().add_to(fmap)
fmap.save('/var/www/html/predict6.html')  # 存成 final.html


# In[ ]:





# In[74]:


a


# In[ ]:





# In[ ]:





# In[ ]:




