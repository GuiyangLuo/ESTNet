import torch
import numpy as np
import pandas as pd
import argparse
import time
import sys,os
sys.path.append('/data/lgy/stPrefiction/codes/FG_SCN/src')
print(sys.path)
import datasets.generate_training_data  as process
import utils.util  as util
import matplotlib.pyplot as plt
from utils.engine import trainer
import os
import torch.nn as nn
from datetime import datetime
from tensorboardX  import SummaryWriter
from statsmodels.tsa.arima_model import ARIMA

from multiprocessing import Process

from sklearn.svm import SVR


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    data = process.preprocess_traffic(data, 5)
    data1 = np.mat(data)
    train_size = int(time_len * rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: pre_len])
        trainY.append(a[pre_len: seq_len +pre_len ])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: pre_len])
        testY.append(b[pre_len: seq_len + pre_len])
    return trainX, trainY, testX, testY


path = '/data/wyt/map_datas/beijing/3/out/result_speeds.csv'

# path = '/data/wyt/map_datas/chengdu/0/out/result_speeds.csv'
#
# path = '/data/wyt/map_datas/xian/0/out/result_speeds.csv'

data = pd.read_csv(path,  index_col=0).fillna(0)
data = data.T
data = np.array(data.values)
data = data[:, 100:101]
print("Original -- ", data.shape)
time_len = data.shape[0]
num_nodes = data.shape[1]
## data shape is : (seq_total_len, num_nodes)
train_rate = 0.8
seq_len = 4
pre_len = 12
x_train,y_train,x_test,y_test = preprocess_data(data, time_len, train_rate, seq_len, pre_len)
x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
print(" trainX = ", x_train.shape,y_train.shape,  x_test.shape, y_test.shape)



def train_baselines():


    x_test_pred = baseline_SVR(x_train, y_train, x_test,y_test)


    print("------------",y_test.shape,  x_test_pred.shape, )
    print(f" results for  {path} for algorithm  baseline_SVR ")
    for seq in range(seq_len):
        metrics =evaluate_metric(y_test[:,seq,:], x_test_pred[:,seq,:])
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.2f}, Test MAPE: {:.2f}, Test RMSE: {:.2f}'
        print(log.format(seq, metrics[0], metrics[1], metrics[2]))

    print("------------", y_test.shape, y_test.shape, )
    x_test_pred = baseline_ARIMA(x_train, y_train, x_test,y_test)

    print(f" results for  {path} for algorithm  baseline_ARIMA")
    for seq in range(seq_len):
        metrics = evaluate_metric(y_test[:, seq, :], x_test_pred[:, seq, :])
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.2f}, Test MAPE: {:.2f}, Test RMSE: {:.2f}'
        print(log.format(seq, metrics[0], metrics[1], metrics[2]))




def baseline_ARIMA(x_train, y_train, x_test_in, y_test_in):
    number_edges = x_test.shape[2]
    number_timeslots = x_test.shape[0]

    pred_results = []
    for edge in range(number_edges):
        print("Processing edge = ",edge )
        batch_result = []
        for batch in range(number_timeslots):


            x_test_edge = x_test[batch, :, edge ]
            x_test_edge = np.squeeze(x_test_edge)

            # print("y_test ", y_test.shape)
            y_test_edge = x_test_in[batch, :, edge ]
            y_test_edge = np.squeeze(y_test_edge)
            # plot_data = list(x_test_edge)
            #
            # plot_data.extend(list( y_test_edge))
            # print("plot_data", len(plot_data))
            # plt.plot(list(range(len(plot_data))),plot_data)
            # plt.show()

            try:
                model = ARIMA(x_test_edge, order=(1, 0, 0))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast(steps=4)[0]
            except:
                output = y_test_edge
                print("except!!!")


            # print(" out put = ", output)
            batch_result.append(output)

        pred_results.append(batch_result)

    pred_results = np.array(pred_results)
    pred_results = pred_results.transpose( 1,2,0 )

    return y_test_in,pred_results




def baseline_SVR(x_train, y_train, x_test, y_test):
    number_edges = x_train.shape[2]
    number_timeslots = x_train.shape[0]

    print("x_train x_test shape is ", x_train.shape,x_test.shape)
    pred_results = []

    for edge in range(number_edges):
        print("Processing edge = ", edge)
        x_train_edge = x_train[:, :, edge, ]
        x_train_edge = np.squeeze(x_train_edge)
        y_train_edge = y_train[:, :, edge, ]
        y_train_edge = np.squeeze(y_train_edge)
        x_test_edge = x_test[:, :, edge, ]
        x_test_edge = np.squeeze(x_test_edge)
        seq_result = []
        for seq in range(seq_len):
            svr_model = SVR(kernel='poly')
            y_train_seq = y_train_edge[:,seq]

            svr_model.fit(x_train_edge, y_train_seq)
            pre = svr_model.predict(x_test_edge)
            seq_result.append(pre)
        seq_result = np.array(seq_result)
        pred_results.append(seq_result.reshape([-1,seq_len]))

    pred_results = np.array(pred_results)
    pred_results = pred_results.transpose( 1,2,0  )

    return y_test, pred_results



def evaluate_metric(pred,real):
    """
    :param pred: (batch,seq_len,#edges,indim)
    :param real: (batch,seq_len,#edges,indim)
    :return:
    """

    device = torch.device('cuda:1')
    pred = torch.Tensor(pred).to(device)
    real = torch.Tensor(real).to(device)
    metrics = util.metric(pred,real )
    return metrics

if __name__ == '__main__':
    train_baselines()