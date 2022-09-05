import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold

from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
from boruta_selector import Boruta

cmap = ListedColormap(["green", "red", "darkorange", "lightseagreen"])

def check(col, col_bor, set_usefull_cols):
    if col in set_usefull_cols and col in col_bor:
        return 1
    elif col in set_usefull_cols:
        return 2
    elif col in col_bor:
        return 3
    else:
        return 4

def plot_heatmap(dic_sample, matrix):
    fig, ax = plt.subplots(figsize=(14, 5))

    hm = sns.heatmap(matrix, cbar=False, cmap=cmap, ax=ax, linewidths=0.1, linecolor='k')
    ax.set_xticks([])
    ax.set_xlabel('Original Columns')
    ax.set_yticks(np.arange(0.5, len(dic_sample['frac'])-0.4, 1), dic_sample['frac'])
    ax.set_ylabel('Fraction of the dataframe sampled (frac)')
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_ticks(np.linspace(1,4,9)[1:8][::2],labels=
                ['Selected useful variable', 'Useful variable not selected',
                'Selected useless variable', 'Useless variable not selected'])
    plt.show()

def plot_percentage_time(dic_sample, matrix, X_big, y_big):
    fig, ax = plt.subplots(ncols=2, figsize=(14, 5))
    ax[0].plot(dic_sample['frac'], (np.array(matrix)==1).sum(axis=1)/50, 'green', label='Green/(Green+Red)')
    ax[0].plot(dic_sample['frac'], (np.array(matrix)==3).sum(axis=1)/50, 'darkorange', label='Orange/(Orange+Blue)')
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_ylabel('Percentage of variables')
    ax[0].set_xlabel('Fraction of the dataframe sampled (frac)')
    ax[0].legend()#loc='upper left'

    line1 = ax[1].plot(dic_sample['frac'], dic_sample['time'], 'black', label='Time')
    ax[1].set_ylabel('Time (seconds)')
    ax[1].set_xlabel('Fraction of the dataframe sampled (frac)')
    ax2 = ax[1].twinx()
    line2 =ax2.plot(dic_sample['frac'], dic_sample['performance'], 'darkblue', label='Performance')
    ax2.set_ylabel('Performance (ROCAUC)')
    ax2.hlines(
        cross_validate(
            RandomForestClassifier(random_state=42),
            X_big.iloc[:,:50], y_big, scoring='roc_auc',
            cv=KFold(shuffle=True, random_state=42))['test_score'].mean(),
        0, 1
    )
    ax2.set_ylim(0.5, 1.05)
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0)
    plt.tight_layout()
    plt.show()

def experiment(fracs, n_samples=5000):
    X_big, y_big = \
    make_classification(n_samples=n_samples,
                        n_features=100,
                        n_informative=40,
                        n_redundant=10,
                        n_classes=2,
                        flip_y=0.1,
                        shuffle=False,
                        random_state=42)

    X_big = pd.DataFrame(X_big, columns=[f'column_{i+1}' for i in range(X_big.shape[1])])
    set_usefull_cols = set(X_big.columns[:50])
    dic_sample = {
        'frac':[],
        'cols_boruta':[],
        'time':[],
        'performance':[]
    }

    for frac in tqdm(fracs):
        X_sample = X_big.sample(frac=frac, replace=False, random_state=42)
        y_sample = y_big[X_sample.index]
        time_start = time()
        boruta_selector = (
            Boruta(n_estimators=100, include_support_weak=True, random_state=42)
            .fit(X_sample, y_sample))

        dic_sample['time'].append(time()-time_start)
        dic_sample['frac'].append(frac)
        dic_sample['cols_boruta'].append(boruta_selector.get_feature_names_out())
        dic_sample['performance'].append(
            cross_validate(
                RandomForestClassifier(random_state=42),
                boruta_selector.transform(X_big), y_big, scoring='roc_auc',
                cv=KFold(shuffle=True, random_state=42))['test_score'].mean())


    matrix = [[check(col, col_bor, set_usefull_cols) for col in X_big.columns] for col_bor in dic_sample['cols_boruta']]

    return dic_sample, matrix, X_big, y_big