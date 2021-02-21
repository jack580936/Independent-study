import json
import math
import fiona
import folium
import branca
import requests
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Polygon

#======================================================================================================
#idw functuion

def idw(lat , lon , ref_point , ref_number):
    total_out = 0
    total_dis = 0
    # optional normalize
    size = 0.01
    sort_list = []
    # sort distance
    for index,row in ref_point.iterrows():
        ref_lon = row['Longitude']
        ref_lat = row['Latitude']
        distance = math.sqrt(((ref_lon - lon)/size) ** 2 + ((ref_lat - lat)/size) ** 2)
        sort_list.append([ref_lon,ref_lat,row['PM2.5'],distance])
    sort_list = sorted(sort_list,key=lambda l:l[3], reverse=False) # 根據距離近到遠排序
    count = 0
    # top ref_number point 找出距離最近的幾個測站，把1/distance累加
    for s in sort_list: # sort_list : [ref_lon,ref_lat,row['PM2.5'],distance] s[2] : pm2.5 , s[3]:distance
        if count == ref_number:
            break
        count += 1
        total_dis += 1 / s[3] #s[3] = distance 1/總距離 (1/distance 1 + 1/distance 2 + ..... 1/distance n)
    count = 0
    # idw :
    # total out : idw預估的pm2.5 -> sigma(前n個測站的pm2.5 * ((1/distance) / 總距離))
    # 這個測站的pm2.5 * 這個測站多重要（權重）
    # 權重：(1/測站離預估點的距離) / (1/總距離) （總距離：前n個測站的距離倒數的和）
    for s in sort_list:
        if count == ref_number:
		 break
        count += 1
        total_dis += 1 / s[3] #s[3] = distance 1/總距離 (1/distance 1 + 1/distance 2 + ..... 1/distance n)
    count = 0
    # idw :
    # total out : idw預估的pm2.5 -> sigma(前n個測站的pm2.5 * ((1/distance) / 總距離))
    # 這個測站的pm2.5 * 這個測站多重要（權重）
    # 權重：(1/測站離預估點的距離) / (1/總距離) （總距離：前n個測站的距離倒數的和）
    for s in sort_list:
        if count == ref_number:
            break
        count += 1
        total_out += ((1 / s[3]) / total_dis) * s[2] #權重 * pm2.5
    return total_out # 預估的pm2.5

#========================================================================================================
#爬蟲 (環保署測站、時間、風力資訊) + idw

ses = requests.Session()
data1 = ses.get('http://taqm.epb.taichung.gov.tw/TQAMNEWAQITABLE.ASPX') #環保署16筆測站
data1.encoding = 'utf-8'
t = pd.read_html(data1.text)[0]
t.drop(t.iloc[:, 1:21], inplace=True, axis=0)
t.drop(t.iloc[:, 1:279], inplace=True, axis=1)
times = str(t[0])
# print(times[21:36]) 時間
df1 = pd.read_html(data1.text)[1]
df1 = df1.drop([0,1])
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
df1[['Latitude', 'Longitude', 'PM2.5']] = df1[['Latitude', 'Longitude','PM2.5']].astype(float)
#df1  顯示16筆測站
df1.to_csv("/home/hpc/df1.csv")
df1 = pd.read_csv("/home/hpc/df1.csv")

data = ses.get('https://www.cwb.gov.tw/V7/observe/real/ObsC.htm?')
data.encoding = 'utf-8'
wind = pd.read_html(data.text)[0]
wind.drop(wind.iloc[:, 2:4], inplace=True, axis=1)
wind.drop(wind.iloc[:, 4:11], inplace=True, axis=1)
wind = wind.drop([30,32,33])
wind.drop('日照時數',inplace=True,axis=1)
wind.drop(wind.index[:2],inplace=True)
wind.rename(columns={ wind.columns[0]: "Sitename" }, inplace=True)
wind.rename(columns={ wind.columns[1]: "time" }, inplace=True)
wind.rename(columns={ wind.columns[2]: "Wind direction" }, inplace=True)
wind.rename(columns={ wind.columns[3]: "Wind speed" }, inplace=True)
wind['Latitude']=[24.256002,24.15295,24.388594,24.276061,24.248517,24.381675,
                  24.363714,24.246428,24.247522,24.347667,24.173142,24.103556,
                  24.272481,24.3593912,24.310436,24.254322,24.092464,24.213106,
                  24.312297,24.347817,24.184536,24.107058,24.179494,24.137006,
                  24.200172,24.215281,24.432739,24.388792,24.322831,24.225972]
wind['Longitude']=[120.5233806,120.572117,121.236339,120.777644,120.903319,121.4203,
                   121.444667,120.833047,121.243669,120.640403,120.722289,120.751061,
                   120.658314,120.5849602,120.729725,120.720692,120.701378,120.703933,
                   120.562242,120.705686,120.528972,120.624103,120.641275,120.637969,
                   120.815806,120.62445,121.303808,121.268675,120.682686,120.800917]
wind.replace('-', 0,inplace=True)
wind.replace('－', '無資料',inplace=True)
wind = wind.reset_index()
#wind 顯示風力資訊
#wind.to_csv("C:\\final_project\\final_project_epa\\wind.csv",encoding="utf_8_sig")
#wind = pd.read_csv("C:\\final_project\\final_project_epa\\wind.csv",encoding="utf_8_sig")

#taichung = gp.read_file("C:\\final_project\\COUNTYID_B.geojson")           #台中邊界
taichungmap_1x1 = gp.read_file("/home/hpc/taichung1x1x1.geojson")         #台中1*1網格
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
# df3 idw point
df3 = pd.DataFrame(columns=['Latitude', 'Longitude', 'PM2.5' , 'Id'])
site_name_count = 1
ref_point_number = 16      # edit here to change ref_number
for i,j,k,l in zip(lat_max,lat_min,lon_max,lon_min):
    site_name = str(site_name_count)
    site_name_count += 1
    df_append = pd.DataFrame([[(i+j)/2 , (k+l)/2 , idw((i+j)/2 , (k+l)/2 , df1 , ref_point_number) , site_name]] ,columns=['Latitude', 'Longitude', 'PM2.5' , 'Id'])
    df3 = df3.append(df_append)
df3.to_csv("/home/hpc/all_point_data_epa.csv")
all_point_data_epa = pd.read_csv("/home/hpc/all_point_data_epa.csv")
lon=list(all_point_data_epa['Longitude'])
lat=list(all_point_data_epa['Latitude'])
all_point_data_epa['Id']=0
Id=taichungmap_1x1['Id']
ans_Id=all_point_data_epa['Id']
# fit id
for i in range(2449):
    ans_Id[i] = i+1
# for d in (range(100)):      #測試資料筆數
#      for m in (range(50)):  #找是在哪列
#             if lon_min[list1[m]-1]<lon[d] and lon[d]<lon_max[list2[m]-1] and lat_min[list1[m]-1]<lat[d] and lat[d]<lat_max[list2[m]-1]:
#                 for n in range(list1[m],list2[m]+1): #找是在該列的哪個
#                     if lon_min[n-1]<lon[d] and lon[d]<lon_max[n-1] and lat_min[n-1]<lat[d] and lat[d]<lat_max[n-1]:
#                         ans_Id[d]=Id[n]-1
taichungmap_1x1 = taichungmap_1x1.merge(all_point_data_epa, on='Id')
taichungmap_1x1['PM2.5']=taichungmap_1x1['PM2.5'].round()

#=============================================================================================================
#folium

fmap=folium.Map(location=[24.2,120.9], zoom_start=10.5)
fmap.choropleth(
                geo_data=taichungmap_1x1,
                name='pm2.5',
                columns=['Id', 'PM2.5'],
                key_on='feature.properties.Id',
 data=all_point_data_epa,
                #threshold_scale=[],
                fill_color='BuGn',
                legend_name='pm2.5',
                line_opacity=0.5,
                fill_opacity=0.8
                )

#fmap.choropleth(
 #               geo_data=taichung,
 #               name='taichung',
 #               line_opacity=0.5,
 #               fill_opacity=0
#                )

folium.GeoJson(
               taichungmap_1x1,
               name='touch',
               style_function=lambda x: {"weight":1, 'color':'#00000000'},
               highlight_function=lambda x: {'weight':3, 'color':'black'},
               tooltip=folium.GeoJsonTooltip(fields=['Id','PM2.5'],aliases=['Id','PM2.5'],labels=True,sticky=True)
              ).add_to(fmap)
#環保署 logo
epa_icon_url = 'https://www.epa.gov.tw/public/MMO/epa/Epa_Logo_01_LOGO.gif'
#風力圖標 logo (十六方位)
wind_icon_url=[ 'https://www.cwb.gov.tw/V7/images/wind_icon/N.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/NNE.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/NE.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/ENE.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/E.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/ESE.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/SE.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/SSE.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/S.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/SSW.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/SW.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/WSW.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/W.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/WNW.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/NW.gif',
                'https://www.cwb.gov.tw/V7/images/wind_icon/NNW.gif']

station=folium.FeatureGroup(name="環保署",show = False)
for i in(range(16)):
    station.add_child(
folium.Marker(
                                        location=[df1['Latitude'][i],df1['Longitude'][i]],
                                        popup=("<b>NAME:</b> {NAME}<br>""<b>PM2.5:</b> {PM25}<br>""<b>TIME:</b> {TIME}<br>")
                                                .format(NAME=str(df1['SiteName'][i]),PM25=str(df1['PM2.5'][i]),TIME=str(times[21:36])),
                                        icon=folium.CustomIcon(epa_icon_url,icon_size=(23,23))  # Creating a custom Icon
                                      )
                      )

station2=folium.FeatureGroup(name="風力",show = False)
for j in(range(30)):
            if wind['Wind direction'][j]=='北':
                 x=0;
            elif wind['Wind direction'][j]=='北北東':
                 x=1;
            elif wind['Wind direction'][j]=='東北':
                 x=2;
            elif wind['Wind direction'][j]=='東北東':
                 x=3;
            elif wind['Wind direction'][j]=='東':
                 x=4;
            elif wind['Wind direction'][j]=='東南東':
                 x=5;
            elif wind['Wind direction'][j]=='東南':
                x=6;
            elif wind['Wind direction'][j]=='南南東':
                 x=7;
            elif wind['Wind direction'][j]=='南':
                 x=8;
            elif wind['Wind direction'][j]=='南南西':
                 x=9;
            elif wind['Wind direction'][j]=='西南':
                 x=10;
            elif wind['Wind direction'][j]=='西南西':
                 x=11;
            elif wind['Wind direction'][j]=='西':
                 x=12;
            elif wind['Wind direction'][j]=='西北西':
                 x=13;
            elif wind['Wind direction'][j]=='西北':
                 x=14;
            elif wind['Wind direction'][j]=='北北西':
                 x=15;
            elif wind['Wind direction'][j]=='靜風':
                 x=0;

            station2.add_child(folium.Marker(
                                                location=[wind['Latitude'][j],wind['Longitude'][j]],
                                                popup=("<b>NAME:</b> {NAME}<br>""<b>Wind speed:</b> {Windspeed}<br>""<b>Wind direction:</b> {winddirection}<br>""<b>TIME:</b> {TIME}<br>")
                                                        .format(NAME=str(wind['Sitename'][j]),Windspeed=str(wind['Wind speed'][j]),winddirection=str(wind['Wind direction'][j]),TIME=str(tim$
                                                icon=folium.CustomIcon(wind_icon_url[x],icon_size=(27,27))
                                              )
                              )

fmap.add_child(station)
fmap.add_child(station2)
folium.LayerControl().add_to(fmap)
# lat/lon to map
# folium.LatLngPopup().add_to(fmap)
fmap.save('/var/www/html/epa2') #存成 final.html

