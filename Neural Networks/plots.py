import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm

MODEL = 'PageAN2Conv1D' #Choose from 'CNN', 'BiGRU', 'SentANConv1D', 'SentANConv2D', 'PageANConv1D', 'PageAN2Conv1D', 'DomainAN'
                    # 'DomainAN2Sent', 'DomainAN2Page', 'DomainAN3'
EMBEDDING = 'no embedding'
FILE_PATH = MODEL + EMBEDDING + '.csv'
save_path = os.getcwd() + "\\" + MODEL + EMBEDDING

df = pd.read_csv(os.path.join(save_path, FILE_PATH), header = None)
iterations = df.shape[1]/2 - 1

for i in range(int(iterations)):
    df2 = pd.concat([df[1], df[i*2+3]], axis=1)
    plt.figure(i)
    if df2[i*2+3].nunique() > 3:
        ax = df.plot.scatter(i*2+3, 1)
        X = sm.add_constant(df2[i*2+3])
        model = sm.RLM(df[1],X)
        res = model.fit()
        plt.plot(df[i*2+3], res.fittedvalues, color = 'g')
    else:
        ax = df2.boxplot(by=i*2+3)
        ax.get_figure().suptitle('')
    ax.get_figure().gca().set_xlabel("Parameter Value")
    ax.get_figure().gca().set_ylabel("Loss")
    ax.set_title(str(df[i*2+2][0])[:-2])
    plt.savefig(os.path.join(save_path, str(df[i*2+2][0])[:-2] + '.png'))


