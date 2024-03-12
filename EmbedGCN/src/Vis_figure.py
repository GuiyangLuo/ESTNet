import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
import sys
import yaml
from types import SimpleNamespace as SN
import torch
import  utils.util as util
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import pickle
from runners.train import  train
from runners.test import  test

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
# %config InlineBackend.figure_formats = ['svg']
# 更改字体字号 10.5=五号字
rcParams['font.size']= 25
rcParams['svg.fonttype']='none'
rcParams['font.sans-serif']=['Times New Roman']
rcParams['mathtext.fontset']='stix'
rcParams['axes.grid']=True
rcParams['grid.linestyle']='--'
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'



def plot_with_layers():
    with  open(f"results_layers.pkl", 'rb') as fo:  # 读取pkl文件数据
        results = pickle.load(fo, encoding='bytes')
    print("results = ", results)
    map = []
    mape = []
    rems = []
    xaxis = list(range(6, 18, 2))
    for i in [12, 8, 6, 10, 14, 16]:
        meanv = results[i].mean(axis=0)
        print(xaxis, i, meanv)

        map.append(meanv[0])
        mape.append(meanv[1] * 100)
        rems.append(meanv[2])

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    markersize = 15
    # ffeda0
    # feb24c
    # f03b20
    ax1.plot(xaxis, map, color='#f03b20', lw = 3, markeredgecolor= 'black',label='MAE',marker='o',ms=markersize)
    ax1.plot(xaxis, rems,  '-', color='#ffeda0',lw = 3,   markeredgecolor= 'black',label='RMSE',marker='*',ms=markersize)
    ax1.set_ylabel('')
    plt.xticks([i for i in xaxis], [int(i/2) for i in xaxis])
    ax1.legend(loc=2)
    plt.ylim(ymax=7.5, ymin=2)
    ax2 = ax1.twinx()
    ax2.plot(xaxis, mape, '-', color='#feb24c',  lw = 2, markeredgecolor= 'black',label='MAPE',marker='p',ms=markersize)
    ax2.set_ylabel('%')
    ax2.legend(loc=1)
    plt.ylim(ymax=9, ymin=6)
    plt.grid(False)
    plt.savefig('layers.pdf', bbox_inches='tight', transparent=True,pad_inches = 0 )
    plt.show()

def plot_with_neighbors():
    with  open(f"results_neighbors.pkl", 'rb') as fo:  # 读取pkl文件数据
        results = pickle.load(fo, encoding='bytes')
    print("results = ", results)
    map = []
    mape = []
    rems = []
    xaxis = np.array(list(range(8, 16, 2)))
    for i in xaxis:
        meanv = results[i].mean(axis=0)
        print(xaxis, i, meanv)
        map.append(meanv[0])
        mape.append(meanv[1] * 100)
        rems.append(meanv[2])
    fig = plt.figure(figsize=(8, 6))
    bar_width = 0.2
    ax1 = fig.add_subplot(111)
    xaxis = np.array(list(range(len(map))))
    plt.xticks([i for i in xaxis], list(range(8, 16, 2)))
    linewidth = 1
    # fee6ce
    # fdae6b
    # e6550d
    bar = ax1.bar(xaxis - bar_width, map,bar_width, color="#e6550d",ec='black',lw=linewidth,  label='MAE', hatch='.')
    ax1.bar(xaxis , rems,bar_width,  color='#fdae6b',ec='black',lw=linewidth, label='RMSE',hatch='X')
    ax1.set_ylabel('')
    ax1.legend(loc=2)
    plt.ylim(ymax=8.5, ymin=1)
    # plt.xlabel('Number of Correlated neighbors ($l$)')
    ax2 = ax1.twinx()
    ax2.bar(xaxis + bar_width, mape, bar_width,color='#fee6ce', lw=linewidth,ec='black',label='MAPE',hatch='o')
    ax2.set_ylabel('%')
    plt.ylim(ymax=9, ymin=5)
    plt.grid(False)
    plt.legend(loc='best', frameon=False)
    ax2.legend(loc=1)
    plt.savefig('neighbors.pdf', bbox_inches='tight', transparent=True,pad_inches = 0 )
    plt.show()



if __name__ == '__main__':
    plot_with_layers()
    plot_with_neighbors()
