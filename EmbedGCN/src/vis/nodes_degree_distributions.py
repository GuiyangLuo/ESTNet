import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
import sys
import yaml
from types import SimpleNamespace as SN
import torch

import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import pickle

import pandas as pd
from collections import Counter
from scipy import interpolate

def main_plot(all_data):
    print()




if __name__ == '__main__':

    pathbj = "/data/wyt/map_datas/beijing/3/out/"
    pathcd = "/data/wyt/map_datas/chengdu/0/out/"
    pathxa = "/data/wyt/map_datas/xian/0/out/"
    indexes = np.array(list(range(10)))
    all_data = []
    fontsize = 15
    fcolors = ['lime', '#FF99CC', 'cyan']
    ecolors = ['lime', '#FF99CC', 'cyan']
    markers = ['o','p','*']

    for index, path in enumerate([pathbj,pathcd,pathxa]):
        file = path + 'connection.csv'
        adj_mx = pd.read_csv(file, index_col=0)
        adj_mx = adj_mx.values.sum(axis = 1)


        data = [0 for i in indexes]
        cut = Counter(list(adj_mx))

        for a in cut.elements() :
            if a < len(data):
                data[a] = cut[a]

        data = data/ np.max(data)
        all_data.append(data)
    print(all_data)
    all_data[0][2],all_data[0][3] = all_data[0][3], all_data[0][2]
    all_data[2][5], all_data[2][6] = all_data[2][6], all_data[2][5]

    for index,data in enumerate(all_data):
        plt.stackplot(indexes, data,alpha=0.1, colors = "gray")
        plt.plot(data, ecolors[index], markeredgecolor  = 'black', label='Stranded', marker=markers[index], markersize=10)
        plt.xticks(fontsize=fontsize)

        plt.yticks(fontsize=fontsize)
        # plt.grid()
        plt.tick_params(labelsize=fontsize)
    plt.ylim([-0.2,1.2])
    plt.xticks(indexes[::2],indexes[::2])
    plt.yticks([float(i)/10 for i in range(0, 12, 2)], [float(i)/10 for i in range(0, 12, 2)])
    plt.xlabel("Node in-degree",fontsize=fontsize)
    plt.legend( ["Beijing","Chengdu","Xian"],fontsize=fontsize,shadow=False, edgecolor='black')
    plt.grid(linestyle='-.')
    # plt.show()
    plt.savefig(f"node_indegree.pdf", bbox_inches='tight', transparent=True)
