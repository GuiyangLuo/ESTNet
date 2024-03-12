import torch.optim as optim
import torch
import os
import torch.nn as nn

import numpy as np
np.set_printoptions(threshold=1e2)

import scipy.sparse as sp
import  utils.util as util
import model.model as model
import model.model_EA as model_EA
import utils.util as util
import pandas as pd
from scipy.spatial import distance
from scipy.special import softmax

class trainer():
    def __init__(self, scaler,args , writer = None):
        device = torch.device(args.device)
        self.writer = writer
        road_connections = pd.read_csv(os.path.join(args.data_origin_path, args.connections_road_network),
                                           index_col=0).values  # shape (# raods, attributes_1)

        road_connections = road_connections + np.identity(road_connections.shape[-1])
        # print("   self.road_connections = ", self.road_connections.shape)

        road_attributes = pd.read_csv(os.path.join(args.data_origin_path, args.edge_att_road_network),  index_col=0).fillna(value=0).values # shape (# raods, attributes_1)
        print("   self.road_attributes = ",  road_attributes.shape)

        sensor_in_traffic_network = pd.read_csv(os.path.join(args.data_origin_path,args.sensor_in_traffic_network),  index_col=0).fillna(value=0).values # shape (# sensor, attributes_2)
        print("   self.sensor_in_traffic_network = ", sensor_in_traffic_network.shape)


        sensors_index = sensor_in_traffic_network[:,4]

        sensor_pos = road_attributes[sensors_index.astype(np.int),:2]
        distance_cor_mat = distance.cdist(sensor_pos, sensor_pos, 'euclidean')
        self.softmax_cor_mat_gt = distance_cor_mat # softmax(distance_cor_mat,axis=-1)

        self.softmax_cor_mat_gt = torch.Tensor(self.softmax_cor_mat_gt).to(device)
        if args.use_tensorboard and writer is not None:
            writer.add_image('cor_mat_gt',
                                  self.softmax_cor_mat_gt,
                                  global_step=0,
                                  dataformats='HW')

        print("  distance_cor_mat = ",  self.softmax_cor_mat_gt.shape,  )

        road_attributes = util.standardization(road_attributes)
        print("sensors_index = ",sensors_index.shape)

        if args.cur_model == 'model_EA':
            self.model = model_EA.EmbedGCN(args, road_connections, road_attributes , sensors_index , writer = writer)
        elif args.cur_model == 'model':
            self.model = model.EmbedGCN(args, road_connections, road_attributes, sensors_index, writer=writer)


        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps = 1.0e-8, amsgrad = False)
        self.args = args
        # learning rate decay
        self.lr_scheduler = None
        if args.lr_decay:
            print('Applying learning rate decay.')
            lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer ,
                                                                milestones=lr_decay_steps,
                                                                gamma=args.lr_decay_rate)

        self.loss =  util.masked_mae ##  self.marginal_loss
        self.scaler = scaler
        self.clip = 5

    def StepLR(self):
        if self.args.lr_decay:
            self.lr_scheduler.step()

    def distance_correlation_loss(self, adjacency_mat):
       ##  self.softmax_cor_mat_gt, pred
        # print(" distance_correlation_loss  ", self.softmax_cor_mat_gt.type(), adjacency_mat.type())
        loss = (self.softmax_cor_mat_gt - adjacency_mat).pow(2)
        loss = loss.mean()
        return loss

    def train(self, input, real_val):
        """
        :param input:  shape (batch_size, in_dim, #edges, seq_len)
        :param real_val: shape (batch_size, #edges, seq_len)
        :return:
        """
        self.model.train()
        self.optimizer.zero_grad()
        output, adjacency_mat = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        # print("haha in engine", predict.shape, real.shape)
        loss_cor = self.distance_correlation_loss(adjacency_mat)*10

        loss = self.loss(predict, real, 0.0)



        (loss ).backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        # print(" engine reture value = ", loss.item(),mape,rmse)
        return loss.item(),mape,rmse



    def eval(self, input, real_val):

        self.model.eval()
        output,_  = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse



