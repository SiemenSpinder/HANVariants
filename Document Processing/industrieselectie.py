import pandas as pd

path = r'C:\Users\sieme\Documents\CBS\Milieurekeningen\Select_VulWebPDAS\Select_VulWebPDAS.xlsx'

df = pd.read_excel(path)

df['Hoofd'] = df['SbicodeActueel']//10000

df_ind = df.loc[df['Hoofd'].isin(set(range(10,34)))]

df_ind = df_ind.dropna()

new_path = r'C:\Users\sieme\Documents\CBS\IndustrieSelectie.xlsx'

df_ind.to_excel(new_path, index=False)
