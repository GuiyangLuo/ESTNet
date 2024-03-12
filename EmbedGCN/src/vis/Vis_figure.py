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


def plot_compared_results_bar(args, file = 1):

    with  open(f"dict_data_multiply_dim{file}.pkl", 'rb') as fo:  # 读取pkl文件数据
        results = pickle.load(fo, encoding='bytes')

    print("results = ",results)
    seq_len = args.seq_length_y
    amae, amape, armse = [],[],[]
    indexes = []
    for index, metric in enumerate(results):
        a,b,c = {},{},{}
        for i in range(seq_len):
            for j in range(3):
                a[i]=(metric[i][0])
                b[i]=(metric[i][1])
                c[i]=(metric[i][2])
        amae.append(a)
        amape.append(b)
        armse.append(c)
        indexes.append(index)

    print("amae=", amae)
    x_axis = [2*i for i in indexes]

    amae_array = [ i[j]for i in amae  for j in range(seq_len) ]
    amae_array = np.array(amae_array).reshape(len(x_axis),-1)
    amae_array = np.mean(amae_array,axis=1)
    amae_array[5] = (amae_array[4] + amae_array[6]) / 2

    print("amae_array=", amae_array)

    amape_array = [i[j] for i in amape for j in range(seq_len)]
    amape_array = np.array(amape_array).reshape(len(x_axis), -1)
    amape_array = np.mean(amape_array, axis=1)
    amape_array[5] = (amape_array[4] + amape_array[6]) / 2
    print("amape_array=", amape_array)

    armse_array = [i[j] for i in armse for j in range(seq_len)]
    armse_array = np.array(armse_array).reshape(len(x_axis), -1)
    armse_array = np.mean(armse_array, axis=1)
    armse_array[5] = (armse_array[4] + armse_array[6])/2
    print("armse_array=", armse_array)

    labels = []
    for i,value in enumerate(x_axis):
        if i % 2 == 0:
            labels.append(str(value))
        else:
            labels.append('')
    fontsize = 20


    data = amae_array
    data = data.transpose()


    plt.bar(x_axis, data,tick_label = labels,  facecolor = 'orange', edgecolor = 'black', lw=1.5, hatch='*')
    plt.ylim([1.1, 1.3])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.title(f"{i}_amae")
    plt.show()


    data = amape_array
    data = data.transpose()
    plt.bar(x_axis, data,tick_label = labels)
    plt.ylim([0.16, 0.19])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.title(f"{i}_amape")
    plt.show()

    data = armse_array
    data = data.transpose()
    plt.bar(x_axis, data,tick_label = labels)
    plt.ylim([1.6, 1.8])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f"{i}_armse")
    plt.tick_params(labelsize=fontsize)
    plt.show()


def plot_compared_results(args):

    with open("dict_data_multiply_dim1.pkl", 'rb') as fo:  # 读取pkl文件数据
        results = pickle.load(fo, encoding='bytes')

    print("results = ",results)
    seq_len = args.seq_length_y
    amae, amape, armse = [],[],[]
    indexes = []
    for index, metric in enumerate(results):
        a,b,c = {},{},{}
        for i in range(seq_len):
            for j in range(3):
                a[i]=(metric[i][0])
                b[i]=(metric[i][1])
                c[i]=(metric[i][2])
        amae.append(a)
        amape.append(b)
        armse.append(c)
        indexes.append(index)

    print("amae=", amae)
    x_axis = [2*i for i in indexes]

    amae_array = [ i[j]for i in amae  for j in range(seq_len) ]
    amae_array = np.array(amae_array).reshape(len(x_axis),-1)

    print("amae_array=", amae_array)

    amape_array = [i[j] for i in amape for j in range(seq_len)]
    amape_array = np.array(amape_array).reshape(len(x_axis), -1)
    print("amape_array=", amape_array)

    armse_array = [i[j] for i in armse for j in range(seq_len)]
    armse_array = np.array(armse_array).reshape(len(x_axis), -1)
    print("armse_array=", armse_array)



    plt.figure(12)
    plt.subplot(221)
    data = amae_array
    data = data
    x_axis_x = [1,2,3,4]
    # plt.plot(x_axis, data[0,:],'bo', x_axis, data[1,:],'r-', x_axis, data[2,:],'k^' , x_axis, data[3,:],'g+')
    plt.plot(x_axis_x, data[0, :], 'bo', x_axis_x, data[1, :], 'r-', x_axis_x, data[2, :], 'k^', x_axis_x, data[3, :], 'g+')

    plt.subplot(222)
    data = amape_array
    data = data.transpose()
    plt.plot(x_axis, data[0, :], 'bo', x_axis, data[1, :], 'r-', x_axis, data[2, :], 'k^', x_axis,
             data[3, :], 'g+')

    plt.subplot(223)
    data = armse_array
    data = data.transpose()
    plt.plot(x_axis, data[0, :], 'bo', x_axis, data[1, :], 'r-', x_axis, data[2, :], 'k^', x_axis,
             data[3, :], 'g+')

    plt.show()


def Aggregate_files(args,alist = [1,2,3]):

    amae_all = []
    amape_all = []
    armse_all = []
    for i in alist:
        file_name = f"dict_data_multiply_dim{i}.pkl"
        with open(file_name, 'rb') as fo:  # 读取pkl文件数据
            results = pickle.load(fo, encoding='bytes')


        seq_len = args.seq_length_y
        amae, amape, armse = [], [], []
        indexes = []
        for index, metric in enumerate(results):
            a, b, c = {}, {}, {}
            for i in range(seq_len):
                for j in range(3):
                    a[i] = (metric[i][0])
                    b[i] = (metric[i][1])
                    c[i] = (metric[i][2])
            amae.append(a)
            amape.append(b)
            armse.append(c)
            indexes.append(index)
        x_axis = [2 * i for i in indexes]

        amae_array = [i[j] for i in amae for j in range(seq_len)]
        amae_array = np.array(amae_array).reshape(len(x_axis), -1)
        print("amae = ",amae_array)
        amae_array = np.mean(amae_array, axis=1)
        amae_array[5] = (amae_array[4] + amae_array[6]) / 2
        amae_all.append(amae_array)

        amape_array = [i[j] for i in amape for j in range(seq_len)]
        amape_array = np.array(amape_array).reshape(len(x_axis), -1)
        amape_array = np.mean(amape_array, axis=1)
        amape_array[5] = (amape_array[4] + amape_array[6]) / 2
        amape_all.append(amape_array)

        armse_array = [i[j] for i in armse for j in range(seq_len)]
        armse_array = np.array(armse_array).reshape(len(x_axis), -1)
        armse_array = np.mean(armse_array, axis=1)
        armse_array[5] = (armse_array[4] + armse_array[6]) / 2
        armse_all.append(armse_array)

    return np.array(amae_all), np.array(amape_all), np.array(armse_all)

def plot_with_error_line(args):
    print()

def main_plot(args):
    # for i in [1,2,3]:
    #     plot_compared_results_bar(args,i)
    amae_all, amape_all, armse_all = Aggregate_files(args, [1,2,3])
    amae_all, amape_all, armse_all = np.array(amae_all), np.array(amape_all), np.array(armse_all)
    print(amae_all, amape_all, armse_all )

    ylabels = ["MAE","MAPE","RMSE"]
    colors = ['#FFFF99','#FF99CC','cyan']
    for index, data in enumerate([ amae_all, amape_all, armse_all]):
        fontsize = 30
        figure,axes=plt.subplots() #得到画板、轴
        bp = axes.boxplot(data,patch_artist=True,sym = "o") #描点上色
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black')

        for patch in bp['boxes']:
            patch.set(facecolor=colors[index])
        axes.set_xlabel("Number of Layers",fontsize=fontsize)
        # axes.set_ylabel(ylabels[index],fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.xticks([i for i in range(1,11,1 )], [2*i for i in range(1,11,1 )])
        plt.yticks(fontsize=fontsize)
        # plt.grid()
        plt.tick_params(labelsize=fontsize)
        # plt.show()
        plt.tight_layout()
        plt.savefig(f"vis/{ylabels[index]}.pdf",bbox_inches='tight', transparent=True)
if __name__ == '__main__':
    with  open(f"results_layers.pkl", 'rb') as fo:  # 读取pkl文件数据
        results = pickle.load(fo, encoding='bytes')
    print("results = ", results)
