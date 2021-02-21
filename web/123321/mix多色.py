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
        total_out += ((1 / s[3]) / total_dis) * s[2] #權重 * pm2.5
    return total_out # 預估的pm2.5

#=======================================================================================================
#爬蟲 (環保署測站、空氣盒子、時間、風力資訊、台中各區天氣) + idw

ses = requests.Session()
data1 = ses.get('http://taqm.epb.taichung.gov.tw/TQAMNEWAQITABLE.ASPX') #16筆測站
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
data2 = requests.get(url='https://aqi.thu.edu.tw/echarts/getairboxobservation') #84筆空氣盒子
js = json.loads(data2.text)
df2 = pd.DataFrame(js)
df2 = df2.replace('─','0')
cols2 = [1,5]
df2 = df2.drop(df2.columns[cols2],axis=1)
df2.rename(columns={ df2.columns[0]: "PM2.5" }, inplace=True)
df2.rename(columns={ df2.columns[1]: "Latitude" }, inplace=True)
df2.rename(columns={ df2.columns[2]: "Longitude" }, inplace=True)
df2.rename(columns={ df2.columns[3]: "SiteName" }, inplace=True)
#df2 顯示84筆空氣盒子
frames = [df1, df2]
final_df = pd.concat(frames)
final_df[['Latitude', 'Longitude', 'PM2.5']] = final_df[['Latitude', 'Longitude','PM2.5']].astype(float)
#final_df 顯示合併後的100筆資料
final_df.to_csv("/home/hpc/final_df.csv") #final_df存成csv檔又命名final_df
final_df = pd.read_csv("/home/hpc/final_df.csv")

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
weather_url=['https://weather.com/zh-TW/weather/hourbyhour/l/aa3b16242a7a1dd982f70522664d004bed7306cf97147c012da8c7cab3c88eaa',
             'https://weather.com/zh-TW/weather/hourbyhour/l/961be9b2c071ce4bd7152bb5d8012612eccb18a230bd7c382b7ebf324b14a377',
             'https://weather.com/zh-TW/weather/hourbyhour/l/c1046cffcf95c30aa14df4dff79010072a1664a2260476433daee2c950e42f60',
             'https://weather.com/zh-TW/weather/hourbyhour/l/c1046cffcf95c30aa14df4dff79010072a1664a2260476433daee2c950e42f60',
             'https://weather.com/zh-TW/weather/hourbyhour/l/e343713b2aa94615ab8ebf55e622879075b79e4d530b0a33ad5251d29800cc99',
             'https://weather.com/zh-TW/weather/hourbyhour/l/b317c7829eecb53b4a74ebdd64a596d9092ba44380830ac8beba971ad29a7a03',
             'https://weather.com/zh-TW/weather/hourbyhour/l/a38f5cfa1883be7020977f4d2a8ac88c8acd9dd362dd992d4fcac42dde1c5837',
             'https://weather.com/zh-TW/weather/hourbyhour/l/ec15f3fe2e4f41951d0c9ad978e33d9d3a703fd8ec63093aed15774b72a68c41',
             'https://weather.com/zh-TW/weather/hourbyhour/l/bf3f7a327eb1f9b80ad96fc5a6661c0c6db64ac66ccc399bf5dbb142dfbba4fa',
             'https://weather.com/zh-TW/weather/hourbyhour/l/436ec7f5df22ab0f5477513766019f94e6ebfedb07af6729a34b1ce3700ec48b',
             'https://weather.com/zh-TW/weather/hourbyhour/l/45c4a53df7df0a517e6789755304be95f0f43a62c22dbd8cbaff6f5e3f048fdb',
             'https://weather.com/zh-TW/weather/hourbyhour/l/ac8734c51232256f170976792c547302e5e5a90587296c2d36042bbd57325c62',
             'https://weather.com/zh-TW/weather/hourbyhour/l/314e852391f0f5de8cc4e78477a18a7c7248c3c08c9d2fa0383865337c1f1230',
             'https://weather.com/zh-TW/weather/hourbyhour/l/ccf14b9b4585a3d73bdad27b3e3e5c0bd2b283f17dcb927c69bb128dea745b81',
             'https://weather.com/zh-TW/weather/hourbyhour/l/d66b396a5ce75bff5d1baa5afa3f0d92b12cacda8b0903d3de677aa4e18f7056',
             'https://weather.com/zh-TW/weather/hourbyhour/l/2a87bb54d883648933d7ddc2e61ebeca48e610aa3c5ae9075dd9b22b0324ad93',
             'https://weather.com/zh-TW/weather/hourbyhour/l/df71fc74476cce5a0a65024728003ba747037661829bdd229a63346360e887a1fd85f40c9daf793132aa39c79c7281cb',
             'https://weather.com/zh-TW/weather/hourbyhour/l/b74ff591fed79ad15708cd03820de2c0f17bff6cc164dac848a0e23b547ee93c',
             'https://weather.com/zh-TW/weather/hourbyhour/l/5363a52af348a3a1dfdb7578d537ab6764aeb5be5f01693c9bad7f0c6e88ae48',
             'https://weather.com/zh-TW/weather/hourbyhour/l/bf4f1b67ced3a5080340a2ff1504c7ffed6438403cc3192b7f28e42a3aaa46e5',
             'https://weather.com/zh-TW/weather/hourbyhour/l/6f4c8de17189181eb9966dc43dcf238289dadb57e0138f3528c48284212ec7cd',
             'https://weather.com/zh-TW/weather/hourbyhour/l/c1fdfe45c6c67cd7b7c8124a0cfd1df50e031f6953274dc0e08e4d0b57769cb2',
             'https://weather.com/zh-TW/weather/hourbyhour/l/3c8397f8314efa1556647c501569d86de74fc74cc0c051f46c01840029b58e43',
             'https://weather.com/zh-TW/weather/hourbyhour/l/22e82d72dc42ac414a71bc2d317f5de25099cda5de412f9fa803b8fc388c1252',
             'https://weather.com/zh-TW/weather/hourbyhour/l/8ae11516dd1c351027b98e2a8180901c07754858557e634ff822a2073c3a5066',
             'https://weather.com/zh-TW/weather/hourbyhour/l/7f0b261ae2c22456bac0d5488664bd6e39541aec9d9ca8e38c73174d90c54882',
             'https://weather.com/zh-TW/weather/hourbyhour/l/5daa4581a5a6d5866d582781627bae43faab3acb0dbe4088c6a1c6fca2af67f9',
             'https://weather.com/zh-TW/weather/hourbyhour/l/bfc72da438689d76e6ae4349ef0ba0f7aa20525f7867feef1cdc5b946dbd4a26',
             'https://weather.com/zh-TW/weather/hourbyhour/l/fd1ac69d84f93b5475d78e5259614ad222104c76a379c4bc481ace3295b23482']
district=['龍井區','北屯區','沙鹿區','西屯區','大里區',
          '豐原區','太平區','霧峰區','后里區','南屯區',
          '潭子區','大甲區','烏日區','大肚區','清水區',
          '中區','北區','西區','南區','東區',
          '東勢區','梧棲區','外埔區','大安區','大雅區',
          '石岡區','新社區','神岡區','和平區']

for i in range(29):
    data=ses.get(weather_url[i],verify=False)
    data.encoding = 'utf-8'
    locals()["d%s"%i]= pd.read_html(data.text)[0]
    locals()["d%s"%i].rename(columns={ locals()["d%s"%i].columns[0]: "區名" }, inplace=True)
    locals()["d%s"%i].fillna(value=0, inplace=True)
    locals()["d%s"%i].replace(0,district[i],inplace=True)
    locals()["d%s"%i].drop('Unnamed: 7',inplace=True,axis=1)
    locals()["d%s"%i].rename(columns={ locals()["d%s"%i].columns[1]: "時間" }, inplace=True)
    locals()["d%s"%i].rename(columns={ locals()["d%s"%i].columns[2]: "天氣" }, inplace=True)
    locals()["d%s"%i].rename(columns={ locals()["d%s"%i].columns[3]: "溫度" }, inplace=True)
    locals()["d%s"%i].rename(columns={ locals()["d%s"%i].columns[4]: "體感" }, inplace=True)
    locals()["d%s"%i].rename(columns={ locals()["d%s"%i].columns[5]: "降雨預報" }, inplace=True)
    locals()["d%s"%i].rename(columns={ locals()["d%s"%i].columns[6]: "濕度" }, inplace=True)
    locals()["d%s"%i]= locals()["d%s"%i].drop([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
weather = pd.concat([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d24,d25,d26,d27,d28], axis=0)
weather.fillna(value=0, inplace=True)
#weather

#taichung = gp.read_file("/home/hpc/COUNTYID_B.geojson")
taichungmap_1x1 = gp.read_file("/home/hpc/taichungcity1x1x1.geojson")
taichung_district = gp.read_file("/home/hpc/taichung_district.geojson")
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
# edit here to change ref_number
ref_point_number = 92
for i,j,k,l in zip(lat_max,lat_min,lon_max,lon_min):
    site_name = str(site_name_count)
    site_name_count += 1
    df_append = pd.DataFrame([[(i+j)/2 , (k+l)/2 , idw((i+j)/2 , (k+l)/2 , final_df , ref_point_number) , site_name]] ,columns=['Latitude', 'Longitude', 'PM2.5' , 'Id'])
    df3 = df3.append(df_append)
df3.to_csv("/home/hpc/all_point_data_mix.csv")
all_point_data_mix = pd.read_csv("/home/hpc/all_point_data_mix.csv")
lon=list(all_point_data_mix['Longitude'])
lat=list(all_point_data_mix['Latitude'])
all_point_data_mix['Id']=0
Id=taichungmap_1x1['Id']
ans_Id=all_point_data_mix['Id']
# fit id
for i in range(2449):
    ans_Id[i] = i+1
# for d in (range(100)):      #測試資料筆數
#      for m in (range(50)):  #找是在哪列
#             if lon_min[list1[m]-1]<lon[d] and lon[d]<lon_max[list2[m]-1] and lat_min[list1[m]-1]<lat[d] and lat[d]<lat_max[list2[m]-1]:
#                 for n in range(list1[m],list2[m]+1): #找是在該列的哪個
#                     if lon_min[n-1]<lon[d] and lon[d]<lon_max[n-1] and lat_min[n-1]<lat[d] and lat[d]<lat_max[n-1]:
#                         ans_Id[d]=Id[n]-1
taichungmap_1x1 = taichungmap_1x1.merge(all_point_data_mix, on='Id')
taichungmap_1x1['PM2.5']=taichungmap_1x1['PM2.5'].round()
taichung_district = taichung_district.merge(weather, on='區名')
#taichung_district
#==================================================================================================
#folium

variable  = 'PM2.5'
colorList = ['#98fb98','#00ff00','#32cd32','#ffff00','#ffd700','#ffa500','#ff6347','#ff0000','#ba55d3']
map_color = cm.StepColormap(colorList,index=[0,10,20,30,40,50,60,70,80],vmin=0,vmax=100,caption = 'PM2.5')

fmap=folium.Map(location=[24.2,120.9], zoom_start=10.5)
#fmap.choropleth(
#                geo_data=taichungmap_1x1,
#                name='pm2.5',
#                columns=['Id', 'PM2.5'],
#                key_on='feature.properties.Id',
#                data=all_point_data_mix,
#                #threshold_scale=[],
#                fill_color='BuGn',
#                legend_name='pm2.5',
#                line_opacity=0.5,
#                fill_opacity=0.8
#                )
folium.GeoJson(
                taichungmap_1x1,
                name='PM2.5',
                style_function=lambda x: {
                                                    'fillColor': map_color(x['properties'][variable]),
                                                    'color': 'black',
                                                    'weight': 0.5,
                                                    'fillOpacity': 0.7
                                                },
                highlight_function=lambda x: {'weight':3, 'color':'black'},
               tooltip=folium.GeoJsonTooltip(fields=['Id','PM2.5'],aliases=['Id','PM2.5'],labels=True,sticky=True)

                ).add_to(fmap)
folium.GeoJson(
               taichung_district,
               name='台中各區天氣',
               style_function=lambda x: {"weight":1, 'color':'#5499c7'},
               highlight_function=lambda x: {'weight':3, 'color':'#2471A3'},
               tooltip=folium.GeoJsonTooltip(fields=['區名','天氣','溫度','降雨預報'],aliases=['區名','天氣','溫度','降雨機率'],labels=True,sticky=False),
               show=False
              ).add_to(fmap)
#fmap.choropleth(
 #               geo_data=taichung,
 #               name='taichung',
 #               line_opacity=0.5,
 #               fill_opacity=0
#                )
#環保署 logo
epa_icon_url = 'https://www.epa.gov.tw/public/MMO/epa/Epa_Logo_01_LOGO.gif'
#空氣盒子
airbox_icon_url='https://images-eu.ssl-images-amazon.com/images/I/215B18jnCsL.png'
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
station1=folium.FeatureGroup(name="環保署",show = False)
for i in(range(16)):
    station1.add_child(
                        folium.Marker(
                                        location=[final_df['Latitude'][i],final_df['Longitude'][i]],
                                        popup=("<b>NAME:</b> {NAME}<br>""<b>PM2.5:</b> {PM25}<br>""<b>TIME:</b> {TIME}<br>")
                                                .format(NAME=str(final_df['SiteName'][i]),PM25=str(final_df['PM2.5'][i]),TIME=str(times[21:36])),
                                        icon=folium.CustomIcon(epa_icon_url,icon_size=(23,23))
                                      )
                        )

station2=folium.FeatureGroup(name="空氣盒子",show = False)
for j in range(16,70):
    station2.add_child(
                        folium.Marker(
                                        location=[final_df['Latitude'][j],final_df['Longitude'][j]],
                                        popup=("<b>NAME:</b> {NAME}<br>""<b>PM2.5:</b> {PM25}<br>""<b>TIME:</b> {TIME}<br>")
                                                .format(NAME=str(final_df['SiteName'][j]),PM25=str(final_df['PM2.5'][j]),TIME=str(times[21:36])),
                                        icon=folium.CustomIcon(airbox_icon_url,icon_size=(23,23))
                                        )
                        )

station3=folium.FeatureGroup(name="風力",show = False)
for k in(range(30)):
            if wind['Wind direction'][k]=='北':
                 x=0;
            elif wind['Wind direction'][k]=='北北東':
                 x=1;
            elif wind['Wind direction'][k]=='東北':
                 x=2;
            elif wind['Wind direction'][k]=='東北東':
                 x=3;
            elif wind['Wind direction'][k]=='東':
                 x=4;
            elif wind['Wind direction'][k]=='東南東':
                 x=5;
            elif wind['Wind direction'][k]=='東南':
                 x=6;
            elif wind['Wind direction'][k]=='南南東':
                 x=7;
            elif wind['Wind direction'][k]=='南':
                 x=8;
            elif wind['Wind direction'][k]=='南南西':
                 x=9;
            elif wind['Wind direction'][k]=='西南':
                 x=10;
            elif wind['Wind direction'][k]=='西南西':
                 x=11;
            elif wind['Wind direction'][k]=='西':
                 x=12;
            elif wind['Wind direction'][k]=='西北西':
                 x=13;
            elif wind['Wind direction'][k]=='西北':
                 x=14;
            elif wind['Wind direction'][k]=='北北西':
                 x=15;
            elif wind['Wind direction'][k]=='靜風':
                 x=0;

            station3.add_child(folium.Marker(
                                                location=[wind['Latitude'][k],wind['Longitude'][k]],
                                                popup=("<b>NAME:</b> {NAME}<br>""<b>Wind speed:</b> {Windspeed}<br>""<b>Wind direction:</b> {winddirection}<br>""<b>TIME:</b> {TIME}<br>")
                                                        .format(NAME=str(wind['Sitename'][k]),Windspeed=str(wind['Wind speed'][k]),winddirection=str(wind['Wind direction'][k]),TIME=str(tim$
                                                icon=folium.CustomIcon(wind_icon_url[x],icon_size=(27,27))
                                              )
                              )

fmap.add_child(station1)
fmap.add_child(station2)
fmap.add_child(station3)
fmap.add_child(map_color)
folium.LayerControl().add_to(fmap)
# lat/lon to map
# folium.LatLngPopup().add_to(fmap)
fmap.save('/var/www/html/xx') #存成 final.html
