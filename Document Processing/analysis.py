import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.legend import Legend


save_path = os.getcwd() + "\\" +'analysis'

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
df = pd.read_excel(r'C:\Users\sieme\Documents\CBS\Neural Networks\PredictionsWebsitesIndustrie.xlsx')

temp = df.groupby(['Hoofd','predict']).size()

#temp = temp.groupby(level=0).apply(lambda x: 100*x/float(x.sum()))

plt.style.use('ggplot')

bars = temp.unstack(1).plot.bar(color = ['orangered', 'forestgreen'], figsize = (11, 5))

L = plt.legend()
L.get_texts()[0].set_text('Not Sustainable')
L.get_texts()[1].set_text('Sustainable')
plt.ylabel('% of companies')
plt.xlabel('Activity')
plt.savefig(os.path.join(save_path,'comp_per_act'))

##df['Cbp_WerkzamePersonenActueel'] = np.log(df['Cbp_WerkzamePersonenActueel'] + 1)
##
##df.groupby('predict').Cbp_WerkzamePersonenActueel.plot(kind = 'kde')


