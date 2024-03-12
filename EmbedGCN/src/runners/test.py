import torch
import numpy as np
import argparse
import time
import utils.util  as util
import matplotlib.pyplot as plt
from utils.engine import trainer
import os
import torch.nn as nn
from datetime import datetime
from tensorboardX  import SummaryWriter
import re

def find_best_in_dir(model_dir = './garage'):
    import os
    files = os.listdir(model_dir)
    for file in files:
        if not file.find('final_best')==-1:
            return os.path.join(model_dir, file)
    min = 1000
    min_file = ''
    for file in files:
        cur = re.findall(r"\d+\.?\d*", file)
        cur = float(cur[-1])
        if cur < min:
            min = cur
            min_file = file
    return  os.path.join(model_dir, min_file)



# def normalization(data):
#     seq_len, fea_len= data.shape
#     x = data.reshape(-1)
#     data = (x - np.min(x)) / (np.max(x) - np.min(x))
#     data = data.reshape( seq_len, fea_len)
#     return data

def normalization(data):
    seq_len, fea_len= data.shape
    x = data.reshape(-1)
    data = (x - np.mean(x)) / np.std(x)
    data = data.reshape( seq_len, fea_len)
    return data




def visulize_weights(engine, args,current_seq =0):


    keys_weight = "residual_blocks_4.fgconv_top.fggcn_layers.{}.fggcn_mlp.weight"
    keys_bias = "residual_blocks_4.fgconv_top.fggcn_layers.{}.fggcn_mlp.bias"
    weights = {}
    bias = {}
    kernels = range(0, args.max_allowed_degree )


    for name, parameters in engine.model.named_parameters():

        for i in kernels:

            if  not name.find(keys_weight.format(i))== -1:
                weights[i] = parameters.detach().cpu().numpy()
                # print("weights is ", weights[i])
            if not name.find(keys_bias.format(i)) == -1:
                bias[i] = parameters.detach().cpu().numpy()


    for i in kernels:
        weights[i] = normalization(weights[i])


    plt.close('all')
    fig = plt.figure()


    ax = []
    ax1 = plt.subplot2grid((2, 14), (0, 0), colspan =1)
    ax2 = plt.subplot2grid((2, 14), (0, 1), colspan =3)
    ax3 = plt.subplot2grid((2, 14), (0, 4), colspan =4)
    ax4 = plt.subplot2grid((2, 14), (0, 8), colspan =6)
    ax5 = plt.subplot2grid((2, 14), (1, 0), colspan =2)
    ax6 = plt.subplot2grid((2, 14), (1, 2), colspan =5)
    ax7 = plt.subplot2grid((2, 14), (1, 7), colspan =7)
    ax.append(ax1)
    ax.append(ax2)
    ax.append(ax3)
    ax.append(ax4)
    ax.append(ax5)
    ax.append(ax6)
    ax.append(ax7)
    seq = 4
    print("kernels ",kernels)
    kernels_rank = [0,2,3,5,1,4,6]
    for index, degree in enumerate(kernels_rank):
        example_plot(ax[index],weights[degree][current_seq, :].reshape(degree + 1, seq).transpose(1, 0),degree+1)

    ax[0].tick_params(labelleft=True)
    ax[4].tick_params(labelleft=True)

    plt.sca(ax[0])
    plt.xticks([0], [1])
    plt.sca(ax[-1])
    plt.xticks([0,2,4,6], [1,3,5,7])

    plt.savefig( f'SEQ_{current_seq}pic.png')
    plt.tight_layout()
    plt.show()

plt.rcParams['savefig.facecolor'] = "0.8"

def example_plot(ax, image, degree, fontsize=20):
    plt.sca(ax)
    plt.imshow(image,interpolation='nearest',aspect='auto')
    ax.locator_params(nbins=3)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.yticks([0,1,2,3], [1,2,3,4])
    plt.tick_params(labelsize=fontsize)
    ax.set_title(str(degree), fontsize=fontsize)

    ax.tick_params(labelbottom=True, labelleft=False)
def test(args):

    device = torch.device(args.device)

    dataloader = util.load_dataset(args.data_output_path, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    engine = None
    engine = trainer(scaler, args)


    model_save_dir = args.model_save_dir + str(args.exp_id) + "/"

    model_path = find_best_in_dir(model_save_dir)
    print("Load the best model ", model_path)
    engine.model.load_state_dict(torch.load(model_path))

    # visulize_weights(engine,args,0)
    # visulize_weights(engine, args, 1)
    # visulize_weights(engine, args, 2)
    # visulize_weights(engine, args, 3)
    # return
    trainable_parameters = []
    for name, param in engine.model.named_parameters():
        if param.requires_grad:
            trainable_parameters.append(name)


    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            engine.model.eval()
            preds = engine.model(testx)[0].transpose(1,3)

        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    metric = []

    # print("results for exp_id {}:".format(args.exp_id))
    # for i in range(args.seq_length_y):
    #     pred = scaler.inverse_transform(yhat[:,:,i])
    #     real = realy[:,:,i]
    #
    #     metrics = util.metric(pred,real)
    #     log = 'Evaluate best model on test data for horizon {:2d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    #     print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
    #     metric.append(metrics)
    # average = np.mean( np.array(metric), axis = 0)
    # log = 'Evaluate best model on test data average, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    # print(log.format(average[0], average[1], average[2]))
    # return metric

    print("results for exp_id {}:".format(args.exp_id))
    for i in [2,5,11]:
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]

        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:2d}, Test MAE: {:.2f}, Test MAPE: {:.2f}, Test RMSE: {:.2f}'
        print(log.format(i + 1, metrics[0], 100*metrics[1], metrics[2]))
        metric.append(metrics)
    average = np.mean(np.array(metric), axis=0)
    log = 'Evaluate best model on test data average, Test MAE: {:.2f}, Test MAPE: {:.2f}, Test RMSE: {:.2f}'
    print(log.format(average[0], 100*average[1], average[2]))
    return metric