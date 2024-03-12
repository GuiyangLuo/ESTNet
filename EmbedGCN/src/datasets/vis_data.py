import pandas as pd
import numpy as np
import h5py
import time

import os
import sys
import matplotlib.pyplot as plt
def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

def plot_time_series(times, speed,edge):

    print(len(times), len(speed))
    plt.figure(figsize=(8, 4))

    # plt.plot(times, speed)
    speed = moving_average(speed, 15)

    daybegin = 24 * 12 * 11
    dayend = 24 * 12 * 18 + 1
    times = times[daybegin:dayend]
    speed = speed[daybegin:dayend]

    plt.plot(times, speed)

    min_index,min_value = [],[]

    for i in range(0,dayend - daybegin,24 * 12):
        speed_day = speed[i:i+24 * 12]
        index = np.where(speed_day == np.min(speed_day))
        min_index.append(index[0]+i), min_value.append(speed[index[0]+i])

    fontsize = 25
    for i in range(len(min_index)):
        if list(min_value[i])[0] > 65:
            continue
        value = list(min_index[i])
        time_forindex = times[int(value[0])]
        time_forindex = time_forindex.split(" ")[-1]

        # plt.scatter(list(min_index[i])[0], list(min_value[i])[0], marker='*', s=100, color='r')
        plt.text(list(min_index[i])[0], list(min_value[i])[0],time_forindex ,  verticalalignment = 'top',horizontalalignment = 'center', fontdict={'size': '20', 'color': 'r'})

    # plt.plot(times, moving_average(speed, 15)-100)

    total_days = int((daybegin - dayend) / 24*12)
    y0, y1 = plt.ylim()
    for i in range(daybegin,dayend,24*12 ):

        plt.vlines((times[i - daybegin]), y0, y1, lw=1.5,
                   colors='white',
                   linestyles='--', )
        plt.vlines((times[i - daybegin]), 35, y1, lw=1.5,
                   colors='black',
                   linestyles='--', )

    plt.xticks(fontsize=fontsize)

    plt.yticks(fontsize=fontsize)
    # plt.grid()
    plt.tick_params(labelsize=fontsize)


    plt.xticks(rotation=360)
    ax = plt.gca()
    # print('all axises is ', times)
    day_weeks = ["Mon.","Tue.","Wed.","Thu.","Fri.","Sat.","Sun."]
    indexes = np.array(ax.get_xticks()[::(24 * 12)][:-1]) + 12 * 12
    print("indexes = ", indexes)
    plt.xticks(indexes, day_weeks)
    font3 = {
        'weight': 'normal',
        'size': 35,
    }
    # plt.title(edge, font3)
    # plt.show()
    plt.savefig(f'out/{edge}_sensor.pdf')
    print(f'{edge}pic.png画完')

if __name__ == "__main__":
    file_name = 'data/metr-la.h5'
    df = pd.read_hdf(file_name)
    num_samples, num_nodes = df.shape
    print(df.shape)
    time_inds = []
    data = df.values
    # print("data = ", data[:10,:10])
    for i in df.index.values:
        newtime = str(i).replace(':00.000000000','').replace('2012-','').replace('T',' ')

        time_inds.append(newtime)

    data = np.expand_dims(df.values, axis=-1)
    goodlist = [6, 98,181,169,99] #9,15,29,64,93,98,155,158,176,186]
    # goodlist = [6, 99]  # 9,15,29,64,93,98,155,158,176,186]
    goodlist = [99]
    for i in goodlist:
        plot_time_series(time_inds, list(np.squeeze(data[:,i])),i )