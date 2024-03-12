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
from multiprocessing import Manager
from multiprocessing import Process

from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)



def preprocess_data(data, time_len, rate, seq_len, pre_len):
    data = process.preprocess_traffic(data, 2)
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





def multi_thread(x_train, y_train, x_test,y_test, func):
    manager = Manager()
    # return_list = manager.list() 也可以使用列表list
    return_dict = manager.dict()
    num_nodes = x_test.shape[-1]
    threads = 50
    keys = list(range(num_nodes))
    gap = int(num_nodes/threads)
    all_threads = []
    old = 0
    for i in range(threads):
        if i == threads - 1:
            indexes = keys[old:]
            thread = Process(target=func, args=(x_train[:,:,indexes], y_train[:,:,indexes], x_test[:,:,indexes],y_test[:,:,indexes], i, return_dict))
        else:
            indexes = keys[old:old + gap]
            thread = Process(target=func, args=(x_train[:,:,indexes], y_train[:,:,indexes], x_test[:,:,indexes],y_test[:,:,indexes], i, return_dict))
        print(f" {i} th thread, processing {indexes}")
        all_threads.append(thread)
        thread.start()
        old += gap
    gt = []
    pred = []
    for t in all_threads:
        t.join()

    for data_i in return_dict.keys():
        gt.append(return_dict[data_i][0])
        pred.append(return_dict[data_i][1])
    # for i in range(40):
    #     print("gt test ", gt[i].shape , pred[i].shape )
    gt = np.vstack(gt, )
    pred = np.vstack(pred, )
    return gt,pred


def train_baselines(path):


    data = pd.read_csv(path, index_col=0).fillna(0)
    data = data.T
    data = np.array(data.values)
    data = data[:200, 100:800]
    print("Original -- ", data.shape)
    time_len = data.shape[0]
    num_nodes = data.shape[1]
    ## data shape is : (seq_total_len, num_nodes)
    train_rate = 0.8
    seq_len = 4
    pre_len = 4
    f = open('SVR_ARIMA_results_500.txt', 'a+')
    x_train, y_train, x_test, y_test = preprocess_data(data, time_len, train_rate, seq_len, pre_len)
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    print(" trainX = ", x_train.shape, y_train.shape, x_test.shape, y_test.shape,file=f)
    ####### prepare the datasets

    gt, preds = multi_thread(x_train, y_train, x_test,y_test, baseline_SVR)
    print("------------",gt.shape,  preds.shape, )
    print(f" results for  {path} for algorithm  baseline_SVR ",file=f)
    for seq in range(seq_len):
        metrics =evaluate_metric(gt[:,seq,:], preds[:,seq,:])
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.2f}, Test MAPE: {:.2f}, Test RMSE: {:.2f}'
        print(log.format(seq, metrics[0], metrics[1], metrics[2]),file=f)

    #---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------

    gt, preds = multi_thread(x_train, y_train, x_test, y_test, baseline_ARIMA)
    print("------------", gt.shape, preds.shape, )
    print(f" results for  {path} for algorithm  baseline_ARIMA ",file=f)
    for seq in range(seq_len):
        metrics = evaluate_metric(gt[:, seq, :], preds[:, seq, :])
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.2f}, Test MAPE: {:.2f}, Test RMSE: {:.2f}'
        print(log.format(seq, metrics[0], metrics[1], metrics[2]),file=f)
    f.close()


def baseline_SVR(x_train_in, y_train_in, x_test_in, y_test_in, i, return_dict):
    number_edges = y_train_in.shape[2]
    seq_len = y_train_in.shape[1]

    # print("x_train x_test shape is ", x_train_in.shape,x_test_in.shape)

    pred_results_all = []


    for edge in range(number_edges):
        print("Processing edge = ", edge)
        x_train_edge = x_train_in[:, :, edge, ]
        x_train_edge = np.squeeze(x_train_edge)
        y_train_edge = y_train_in[:, :, edge, ]
        y_train_edge = np.squeeze(y_train_edge)
        x_test_edge = x_test_in[:, :, edge, ]
        x_test_edge = np.squeeze(x_test_edge)
        seq_result = []

        for seq in range(seq_len):
            svr_model = SVR(kernel='poly')
            y_train_seq = y_train_edge[:,seq]

            svr_model.fit(x_train_edge, y_train_seq)
            pre = svr_model.predict(x_test_edge)
            # print(" inside SVR ", x_train_edge.shape, y_train_seq.shape,x_test_edge.shape, pre.shape)

            seq_result.append(pre)

        seq_result = np.array(seq_result)

        pred_results_all.append(seq_result.reshape([-1,seq_len]))

    pred_results = np.array(pred_results_all)
    pred_results = pred_results.transpose( 1,2,0  )
    return_dict[i] = (y_test_in, pred_results)



def baseline_ARIMA(x_train, y_train, x_test_in, y_test_in, i, return_dict):
    number_edges = x_test_in.shape[2]
    number_timeslots = x_test_in.shape[0]
    batch_result_pred_all = []
    batch_result_gt_all = []
    for edge in range(number_edges):
        print("Processing edge = ",edge )
        batch_result_pred = []
        batch_result_gt = []
        for batch in range(number_timeslots):

            x_test_edge = x_test_in[batch, :, edge ]
            x_test_edge = np.squeeze(x_test_edge)

            # print("y_test ", y_test.shape)
            y_test_edge = y_test_in[batch, :, edge ]
            y_test_edge = np.squeeze(y_test_edge)
            # plot_data = list(x_test_edge)
            # plot_data.extend(list( y_test_edge))
            # print("plot_data", len(plot_data))
            # plt.plot(list(range(len(plot_data))),plot_data)
            # plt.show()
            try:
                model = ARIMA(x_test_edge, order=(1, 0, 0))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast(steps=4)[0]
                batch_result_pred.append(output)
                batch_result_gt.append(y_test_edge)
            except:
                None

        batch_result_pred_all.append(batch_result_pred)
        batch_result_gt_all.append(batch_result_gt)

    pred_results = np.array(batch_result_pred_all)
    pred_results = pred_results.transpose( 1,2,0 )

    gt_results = np.array(batch_result_gt_all)
    gt_results = gt_results.transpose(1, 2, 0)

    return_dict[i] = (gt_results,pred_results)




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

    path_bj = '/data/wyt/map_datas/beijing/3/out/result_speeds.csv'
    path_cd = '/data/wyt/map_datas/chengdu/0/out/result_speeds.csv'
    path_xa= '/data/wyt/map_datas/xian/0/out/result_speeds.csv'
    for path in [path_bj,path_xa,path_cd]:
        train_baselines(path)