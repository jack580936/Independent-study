def search(x, y):  #尋找鄉鎮
    global shapes, townnames
    return next((townnames[town_id]  #如果鄉鎮區域包含傳入的經緯度就傳回townnames[town_id]
                 for town_id in shapes  #逐一尋找各鄉鎮
                 if shapes[town_id].contains(Point(x, y))), None)

import fiona
from shapely.geometry import shape, Point
import os

lng = float(input('輸入經度：'))
lat = float(input('輸入緯度：'))
module_dir = os.path.dirname(__file__)  #取得目前目錄
collection = fiona.open(os.path.join(module_dir, 'TOWN_MOI_1070516.shp'))
shapes = {}
townnames = {}

for f in collection:
    town_id = f['properties']['TOWNCODE']  #鄉鎮代碼
    shapes[town_id] = shape(f['geometry'])  #鄉鎮界限經緯度
    townnames[town_id] = f['properties']['COUNTYNAME'] + ',' + f['properties']['TOWNNAME'] #search函式傳回值

print(search(float(lng), float(lat)))
