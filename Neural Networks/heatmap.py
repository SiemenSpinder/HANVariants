#basic
import os
import numpy as np
import pandas as pd

#matplotlib related
import matplotlib.pyplot as plt
import matplotlib.cm
 
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

from mpl_toolkits.basemap import Basemap

#path to predictions
path = r'C:\Users\sieme\Documents\CBS\Neural Networks\PredictionsWebsitesIndustrie.xlsx'
df = pd.read_excel(path)

df.loc[df['PROVINCIE'] == 'Noord Holland', 'PROVINCIE'] = 'Noord-Holland'
df.loc[df['PROVINCIE'] == 'Noord Brabant', 'PROVINCIE'] = 'Noord-Brabant'
df.loc[df['PROVINCIE'] == 'Zuid Holland', 'PROVINCIE'] = 'Zuid-Holland'
df.loc[df['PROVINCIE'] == 'Fryslân', 'PROVINCIE'] = 'Friesland'
df_temp = df
df = df.loc[df['predict'] == 1]

#path to stats provinces
path2 = r'C:\Users\sieme\Documents\CBS\Neural Networks\geo_provincie.xlsx'
provinces = pd.read_excel(path2)

#merge predictions with province stats
df = df.merge(provinces, left_on = 'PROVINCIE', right_on = 'Province')

#########  calculate stats of predictions
#counter = number of companies per capita
#counter2 = absolute number of companies
#counter3 = number of companies per squared meter
#counter4 = discarded
#counter5 = number of companies as percentage of total companies

new_areas = pd.DataFrame({'counter': df['PROVINCIE'].value_counts()/(sorted(provinces.Population.tolist(), reverse = True)),
            'counter2': df['PROVINCIE'].value_counts(),
            'counter3': df['PROVINCIE'].value_counts()/(sorted(provinces.Size.tolist(), reverse = True)),
            'counter5': df['PROVINCIE'].value_counts()/df_temp['PROVINCIE'].value_counts()
                         })
new_areas.index.names = ['PROVINCIE']
new_areas = new_areas.reset_index()
new_areas = new_areas.merge(provinces, left_on = 'PROVINCIE', right_on = 'Province')

#create base empty map
fig, ax = plt.subplots(figsize=(10,20))

m = Basemap(resolution='i', # c, l, i, h, f or None
            projection='merc',
            lat_0=52.2, lon_0=5.2,
            llcrnrlon=3.1, llcrnrlat= 50.7, urcrnrlon=7.4, urcrnrlat=54)

m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.drawcoastlines()


#read shapes of provinces
m.readshapefile('../Neural Networks/gadm36_NLD_shp/gadm36_NLD_1', 'areas', drawbounds = True)

#merge shapes with predictions
df_poly = pd.DataFrame({
        'shapes': [Polygon(np.array(shape), True) for shape in m.areas],
        'area': [area['NAME_1'] for area in m.areas_info]
    })
df_poly = df_poly.merge(new_areas, left_on='area', right_on = 'PROVINCIE')
df_poly = df_poly[~np.isnan(df_poly.counter5)]    

#colour
cmap = plt.get_cmap('YlOrBr')   
pc = PatchCollection(df_poly.shapes, zorder=2)
norm = Normalize()

#set colours based on counter
pc.set_facecolor(cmap(norm(df_poly['counter5'].fillna(0).values)))
ax.add_collection(pc)

mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

mapper.set_array(df_poly['counter5'])
plt.colorbar(mapper, shrink=0.4)

#save map
save_path = os.getcwd() + "\\" +'analysis'

if not os.path.exists(save_path):
    os.makedirs(save_path)

plt.savefig(os.path.join(save_path,'comp_per_area'))

