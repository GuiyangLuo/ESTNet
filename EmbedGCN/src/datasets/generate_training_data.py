from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
from types import SimpleNamespace as SN
import yaml
import matplotlib.pyplot as plt

def moving_average(interval, windowsize):
    """
    平滑
    :param interval:要平滑的list
    :param windowsize: 窗口大小
    :return:
    """
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'valid')
    return re

def plot_time_series( speed, edge):

    times = list(range(0,len(speed)))
    speed = list(speed)
    print(len(times), len(speed))
    plt.figure(figsize=(100, 5))
    plt.scatter(list(range(0,len(speed))), speed, s=2, c='black')


    speed3 = moving_average(speed, 10)
    plt.plot(list(range(0,len(speed3))), speed3, c='g')



    plt.xticks(rotation=360)
    ax = plt.gca()
    # print('all axises is ', times)
    ax.set_xticks(ax.get_xticks()[::(24 * 4)])
    font3 = {
        'weight': 'normal',
        'size': 35,
    }
    plt.grid()
    plt.title(edge, font3)
    plt.show()

    print(f'{edge}pic.png画完')

def preprocess_traffic(dfvalues, windowsize = 1):
    time_steps, edges = dfvalues.shape
    print("shape of dfvalues in preprocess_traffic ", dfvalues.shape)
    for edge in range(edges):
        edge_values = dfvalues[:,edge]
        for index in range(time_steps):
            if np.isnan(edge_values[index]):
                edge_values[index] = 0

    for edge in range(edges):
        edge_values = dfvalues[:,edge]
        for index in range(time_steps):
            if edge_values[index] == 0:
                if index ==0:
                    edge_values[index] = edge_values[index+1]
                elif index==time_steps-1:
                    edge_values[index] = edge_values[index - 1]
                else:
                    edge_values[index] = (edge_values[index - 1] + edge_values[index+1] )/2

    results= []
    for edge in range(edges):
        edge_values = dfvalues[:,edge]
        results.append(moving_average(edge_values,windowsize))

    out = np.array(results).transpose()

    print("shape of dfvalues out preprocess_traffic ", out.shape)
    return out





def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    ## data should be stored in [#edges, time_steps]


    edge = 1
    # plot_time_series(df.values[:, edge], edge)

    df_values = df.values
    edge = 1
    plot_time_series(df.values[:, edge], edge)

    num_samples, num_nodes = df_values.shape
    print("shape of df_values",num_nodes, num_samples)

    data = np.expand_dims(df_values, axis=-1)
    feature_list = [data]

    data = np.concatenate(feature_list, axis=-1)

    x, y = [], []

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)) ) # Exclusive
    print("df.shape",num_samples, df.shape, min_t, max_t,x_offsets, y_offsets, data.shape )
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    print('shape of x and y', x.shape, y.shape)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    filename = os.path.join(args.data_origin_path,args.traffic_sequence_name)

    df = pd.read_hdf(filename)#.fillna(value=0)


    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))
    # x_offsets = [-11 - 10 - 9 - 8 - 7 - 6 - 5 - 4 - 3 - 2 - 1   0]
    # y_offsets = [1  2  3  4  5  6  7  8  9 10 11 12]


    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)

    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets = x_offsets,
        y_offsets = y_offsets
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]
    # numpy : last num_test
    # print('----', list(x_offsets.shape), list( [1]), list(x_offsets.shape) + [1])
    # print('----', x_offsets.shape)
    # print('----',x_offsets.reshape(list(x_offsets.shape) + [1]))

    combinedc = list(zip(x,y))
    np.random.shuffle(combinedc)
    x[:],y[:] = zip(*combinedc)


    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.data_output_path, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def _get_config_dataset(yamlname = 'dataset'):


    with open(os.path.join(os.path.dirname(__file__), "../config", "algs","{}.yaml".format(yamlname)),
              "r") as f:
        try:
            dataset_config_dict = yaml.load(f,Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(yamlname, exc)
    return dataset_config_dict



if __name__ == "__main__":

    dataset_config = _get_config_dataset('metr-la')

    args = SN(**dataset_config)
    print('configures for dataset',args)
    if not os.path.exists(args.data_output_path):
        os.makedirs(args.data_output_path)
    generate_train_val_test(args)


