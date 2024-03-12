import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np
import scipy.sparse as sp
import math
import pandas as pd
import numpy as np
import numpy as np
import cvxpy as cp

class Caculate_parameters_conv():

    def __init__(self, number_neighbors, seq_input_x, max_allow_spatial_conv = 3, max_allow_dilation = 2, weight = 'std'):
        super(Caculate_parameters_conv, self).__init__()
        self.number_neighbors = number_neighbors
        self.seq_input_x = seq_input_x
        self.max_allow_spatial_conv = max_allow_spatial_conv
        self.max_allow_dilation = max_allow_dilation
        self.weight =  weight

    def main(self):
        w = []
        b = []
        v = []
        bags = self.constructed_bags()
        for bag in bags:
            w.append(bag[4])
            b.append(bag[5])
            if self.weight == 'std':
                v.append(- np.array([bag[4],bag[5]]).std())
            elif  self.weight == 'mean':
                v.append(- np.array([bag[4], bag[5]]).mean())

        x_list = self.interger_programming_conv(w,b,v,self.number_neighbors - 1, self.seq_input_x - 1  )
        final_convs = []
        for index,x in enumerate(x_list):
            x = int(x)
            [final_convs.append(bags[index]) for i in range(x)]
        np.random.shuffle(final_convs)
        return final_convs



    def constructed_bags(self):
        bags = []
        for ker1 in range(1,self.max_allow_spatial_conv+1):
            for ker2 in range(1,self.max_allow_spatial_conv+1):
                for dila1 in range(1, self.max_allow_dilation + 1):
                    for dila2 in range(1, self.max_allow_dilation + 1):
                        if ker1 < dila1 or ker2 < dila2:
                            continue
                        weig1 = dila1 * (ker1 - 1)
                        weig2 = dila2 * (ker2 - 1)
                        bag = (ker1, ker2, dila1, dila2, weig1, weig2)
                        bags.append(bag)

        def compare_func(x):
            return  max(x[0],x[1]) * 10 +  min(x[0],x[1])
        bags = sorted(bags, key = compare_func, reverse=True)
        return bags

    def interger_programming_conv(self,w,b, v, neighbors,seq_in):
        n = len(w)

        c = np.array(v)

        a = np.array([w , b]).reshape(2,-1)

        # 输入b值（3×1）
        b = np.array([neighbors,seq_in])

        # 创建x，个数是3
        x = cp.Variable(n, integer=True)

        # 明确目标函数（此时c是3×1，x是3×1,但python里面可以相乘）
        objective = cp.Maximize(cp.sum(c * x))

        # 明确约束条件，其中a是3×3，x是3×1,a*x=b(b为3×1的矩阵)

        constriants = [0 <= x, a * x == b]
        # 求解问题
        prob = cp.Problem(objective, constriants)

        resluts = prob.solve(solver=cp.CPLEX)

        return x.value

class CommonGCN(nn.Module):

    def __init__(self, args, adj_matrix, edge_attribute, sensor_indexes):
        super(CommonGCN, self).__init__()
        self.adj_matrix = adj_matrix
        self.edge_attribute = edge_attribute
        self.sensor_indexes = sensor_indexes
        self.edge_attribute_len = edge_attribute.shape[-1]
        self.edge_att_gcn_module_list = nn.ModuleList()
        self.edge_att_gcn_module_list_activations = nn.ModuleList()
        previous = self.edge_attribute_len
        for index, feature_size in enumerate(args.static_gcn_feature_size):
            block_net = nn.Sequential(nn.Linear(previous, feature_size, bias=True), )
            previous = feature_size
            self.edge_att_gcn_module_list.add_module('edge_att_gcn_{}'.format(index), block_net)
            self.edge_att_gcn_module_list_activations.add_module('edge_att_gcn_activations_{}'.format(index),  nn.ReLU())



    def forward(self,):

        input = self.edge_attribute
        for net,act in zip(self.edge_att_gcn_module_list, self.edge_att_gcn_module_list_activations):
            input = net(input)
            all_nodes_aggregate = torch.spmm(self.adj_matrix, input)
            input = act(input)

        output = input[self.sensor_indexes]

        return output


class Static_features(nn.Module):

    def __init__(self, args, connections_road, edge_attribute, sensor_indexes):
        super(Static_features, self).__init__()

        print(" init  Static_features networks ")
        device = torch.device(args.device)
        edge_attribute = torch.Tensor(edge_attribute).to(device)
        sensor_indexes = torch.LongTensor(sensor_indexes).to(device)

        self.number_of_scales = args.number_of_scales
        self.number_of_roads = connections_road.shape[-1]
        identity_mat = np.identity(self.number_of_roads)
        identity_mat_tensor = torch.Tensor(identity_mat).to(device)
        self.CommonGCN_allscales = nn.ModuleList([CommonGCN(args, identity_mat_tensor, edge_attribute, sensor_indexes)])
        connections_road_coo = sp.coo_matrix( connections_road )
        node_degree = np.array(connections_road_coo.sum(1))



        for i in range(1, self.number_of_scales):
            connections_road_coo =  connections_road_coo.dot(connections_road_coo)
            d_inv_sqrt = np.power(node_degree, -0.5).flatten()
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            adj_matrix_scale = connections_road_coo.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
            adj_matrix_scale = torch.Tensor(adj_matrix_scale).to(device)
            self.CommonGCN_allscales.append(CommonGCN(args, adj_matrix_scale, edge_attribute, sensor_indexes))


    def forward(self, ):
        output = []
        for i in range( self.number_of_scales):
            output.append(self.CommonGCN_allscales[i]())
        output = torch.cat(output, axis=1)
        return output


class Dynamic_features(nn.Module):

    def __init__(self, args):
        super(Dynamic_features, self).__init__()
        # accepted input should be shape of :  [seq_len, batch, input_size]
        self.gru = nn.GRU(args.in_dims, args.dynamic_gru_feature_size, args.gru_number_of_layers, bidirectional  = args.gru_bidirectional)
        self.gru_bidirectional = 1 if args.gru_bidirectional is False else 2
        # print("  self.gru_bidirectional = ", self.gru_bidirectional, )
        self.h_0 =(torch.randn(args.gru_number_of_layers*self.gru_bidirectional, args.number_of_sensors *  args.batch_size, args.dynamic_gru_feature_size))
        device = torch.device(args.device)
        self.h_0 = torch.Tensor(self.h_0).to(device)


    def init_hidden(self):
        for  param  in self.gru.parameters():
            print("name = ",  param )
            nn.init.orthogonal(param)


    def forward(self, input):
        # input shape (batch_size, in_dim, #edges, in_seq)
        batch_size, in_dim, num_sensors, in_seq = input.shape[0], input.shape[1], input.shape[2], input.shape[3]
        input = input.permute(3,0,2,1).contiguous()
        # print(" Dynamic_features  input for gru =  ", input.shape)
        input = input.view(in_seq,-1,in_dim)
        # print(" Dynamic_features  input for gru =  ", input.shape)
        output, hidden_0 = self.gru(input,self.h_0)  # (num_layers * num_directions, batch, hidden_size)

        hidden_0 = hidden_0[-self.gru_bidirectional:,:,:]
        # print(" Dynamic_features =  ", hidden_0.shape)
        hidden_0 = hidden_0.permute(1,0,2).contiguous().view(batch_size,num_sensors,-1).contiguous()
        # print(" Dynamic_features =  ", hidden_0.shape )
        return hidden_0

class Spatial_temporal(nn.Module):

    def __init__(self, args):
        super(Spatial_temporal, self).__init__()
        # self.spatial_temporal_networks = nn.Sequential()
        channels_conv3d = [args.in_dims, 256, 512, 256, 64, 1]

        channels_conv3d1 = 32
        self.spatial_temporal_block_1 = nn.Sequential(
            nn.Conv3d(args.in_dims, channels_conv3d1, kernel_size=(1, 1, 1), bias=True),
            nn.ReLU(),
            nn.BatchNorm3d(channels_conv3d1)
        )


     #    channels_conv3d_2 = 64
     #    self.spatial_temporal_block_temporal = nn.Sequential(
     #        nn.Conv3d(channels_conv3d1, channels_conv3d_2, kernel_size=(1, 1, 3), dilation=(1, 1, 1),  stride = (1, 1, 3), bias=True),
     #        nn.ReLU(),
     #        nn.BatchNorm3d(channels_conv3d_2),
     #        nn.Conv3d(channels_conv3d_2, 12, kernel_size=(1, 1, 4), dilation=(1, 1, 1), bias=True),
     #        nn.ReLU(),
     # )

        channels_conv3d_2 = 64
        self.spatial_temporal_block_temporal = nn.Sequential(
            nn.Conv3d(channels_conv3d1, channels_conv3d_2, kernel_size=(1, args.top_k_neighbors, 12), dilation=(1, 1, 1), stride=(1, 1, 1),
                      bias=True),
            nn.ReLU(),
            nn.Conv3d(channels_conv3d_2, 12, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                      bias=True),
        )

        # kernel_size = 1
        #
        #
        #
        # self.spatial_temporal_block_spatial = nn.Sequential(
        #     nn.BatchNorm3d(channels_conv3d_2),
        #     nn.Conv3d(channels_conv3d_2, 12, kernel_size=(1, kernel_size, 1), bias=True),
        #     nn.ReLU(),
        # )




    def forward(self, threeDinput):
        # input shape (batch_size, number_sensors, neighbors, 1, in-seq)
        threeDinput = threeDinput.permute(0,3,1,2,4).contiguous()
        # print("shape of input ",threeDinput.shape)

        output =  self.spatial_temporal_block_1(threeDinput)
        # print("shape of spatial_temporal_block_temporal", output.shape  )
        output = self.spatial_temporal_block_temporal(output)
        # print("shape of spatial_temporal_block_temporal after", output.shape)

        # output = self.spatial_temporal_block_spatial(output)



        output = output.squeeze(-1)
        # output shape (batch_size, in_seq, #edges, in_dim)
        return output

class Spatial_temporal_Inception_Pure_v1(nn.Module):

    def __init__(self, args):
        super(Spatial_temporal_Inception_Pure_v1, self).__init__()


        args.spatial_temporal_hidden.append(args.seq_length_y)
        increase_dimension_hidden_size = 32
        self.increase_dimension_net = nn.Sequential(nn.Conv3d(1, increase_dimension_hidden_size, kernel_size=(1, 1, 1),
                                            dilation=(1, 1, 1), stride=(1, 1, 1), bias=True), nn.ReLU(),)

        self.inception_out_feature_size = 64

        self.spatial_temporal_net_1n = nn.Sequential(
            nn.BatchNorm3d(increase_dimension_hidden_size),
            nn.Conv3d(increase_dimension_hidden_size, 48,
                      kernel_size=(1, args.top_k_neighbors , 1),
                      bias=True),
            nn.ReLU(),
            nn.Conv3d(48, self.inception_out_feature_size,
                      kernel_size=(1, 1, args.seq_length_y,),
                      bias=True),
            nn.ReLU(),
        )


        neighbors_list, seq_list = args.pooling_neighbors_list, args.pooling_seq_list
        # print("*****",args.max_allow_spatial_conv, args.seq_length_x, args.max_allow_spatial_conv, neighbors_list,seq_list)
        self.spatial_temporal_net_pooling = nn.ModuleList()
        for index, conv_parameters in enumerate(neighbors_list):
            if index == 0:
                norm_3d = nn.BatchNorm3d(increase_dimension_hidden_size)

                pool_net = nn.AvgPool3d(kernel_size=(1, neighbors_list[index], seq_list[index]),
                                        )
                conv3d_net = nn.Conv3d(increase_dimension_hidden_size, self.inception_out_feature_size,
                                       kernel_size=(1, 1, 1), padding_mode='replicate',
                                       bias=True)
            else:
                norm_3d = nn.BatchNorm3d(self.inception_out_feature_size)
                pool_net = nn.AvgPool3d(kernel_size=(1, neighbors_list[index], seq_list[index]),
                                        )
                conv3d_net = nn.Conv3d(self.inception_out_feature_size, self.inception_out_feature_size,
                                       kernel_size=(1, 1, 1), padding_mode='replicate',
                                       bias=True)

            concat_net = nn.Sequential(norm_3d, pool_net, conv3d_net, nn.ReLU())
            print("spatial_temporal_net_pooling", pool_net)
            self.spatial_temporal_net_pooling.add_module('edge_att_gcn_activations_{}'.format(index), concat_net)

        Parameter_Model = Caculate_parameters_conv(args.top_k_neighbors, args.seq_length_x, args.max_allow_spatial_conv,
                                                   args.max_allow_dilation, weight = 'std')
        conv_bags = Parameter_Model.main()
        print("conv_bags = ", conv_bags)
        self.spatial_temporal_net_nn = nn.ModuleList()
        for index, conv_parameters in enumerate(conv_bags):
            if index == 0:
                norm_3d = nn.BatchNorm3d(increase_dimension_hidden_size)
                conv3d_net = nn.Conv3d(increase_dimension_hidden_size, self.inception_out_feature_size,
                                       kernel_size=(1, conv_parameters[0], conv_parameters[1]),
                                       dilation=(1, conv_parameters[2], conv_parameters[3]), padding_mode='replicate',
                                       bias=True)
            else:
                norm_3d = nn.BatchNorm3d(self.inception_out_feature_size)
                conv3d_net = nn.Conv3d(self.inception_out_feature_size, self.inception_out_feature_size,
                                       kernel_size=(1, conv_parameters[0], conv_parameters[1]),
                                       dilation=(1, conv_parameters[2], conv_parameters[3]), padding_mode='replicate',
                                       bias=True)
            concat_net = nn.Sequential(norm_3d, conv3d_net, nn.ReLU())
            print(" spatial_temporal_net_nn ",conv3d_net)
            self.spatial_temporal_net_nn.add_module('edge_att_gcn_activations_{}'.format(index), concat_net)

        self.final_conv= nn.Sequential(
                nn.Conv3d(self.inception_out_feature_size*3, 64, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                          bias=True),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Conv3d(64, 12, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                          bias=True),
            )

    def forward(self, threeDinput):
        # input shape (batch_size, number_sensors, neighbors, 1, in-seq)
        threeDinput = threeDinput.permute(0, 3, 1, 2, 4).contiguous()
        # print("  shape of threeDinput is ", threeDinput.shape)
        threeDinput = self.increase_dimension_net(threeDinput)
        # print(" shaoe of spatial_temporal_net ", threeDinput.shape)

        output_nn = threeDinput.clone()
        output_pooling = threeDinput.clone()
        # print(" shaoe of spatial_temporal_net ", output_pooling.shape)
        for index, net in enumerate(self.spatial_temporal_net_nn):
            output_net = net(output_nn)
            size_neighbor, size_temporal = output_net.shape[3],output_net.shape[4]
            # print(" shaoe of spatial_temporal_net ",output_net.shape, output.shape )
            if index > 0 :
                # print(" shaoe of spatial_temporal_net ", net, size_neighbor, size_temporal,   output[:,:,:,-size_neighbor:,-size_temporal:].shape, output_net.shape)
                output_nn = output_nn[:,:,:,-size_neighbor:,-size_temporal:] + output_net
            else:
                output_nn = output_net
        # print(" shaoe of spatial_temporal_net ",output_pooling.shape )
        for index, net in enumerate(self.spatial_temporal_net_pooling):
            output_net = net(output_pooling)
            size_neighbor, size_temporal = output_net.shape[3], output_net.shape[4]
            # print(" shaoe of spatial_temporal_net ",output_net.shape, output.shape )
            if index > 0:
                # print(" shaoe of spatial_temporal_net ",  output[:,:,:,-size_neighbor:,-size_temporal:] .shape, output_net.shape)
                output_pooling = output_pooling[:, :, :, -size_neighbor:, -size_temporal:] + output_net
            else:
                output_pooling = output_net

        output_1n = self.spatial_temporal_net_1n(threeDinput)
        output = torch.cat([output_pooling, output_nn, output_1n ], dim = 1)


        output = self.final_conv(output)
        output = output.squeeze(-1)
        return output


class Spatial_temporal_Inception_Pure_v2(nn.Module):
    #### Residual connections + sharp hidden size of features,
    def __init__(self, args):
        super(Spatial_temporal_Inception_Pure_v2, self).__init__()
        args.spatial_temporal_hidden.append(args.seq_length_y)
        increase_dimension_hidden_size = 32
        self.increase_dimension_net = nn.Sequential(nn.Conv3d(1, increase_dimension_hidden_size, kernel_size=(1, 1, 1),
                                            dilation=(1, 1, 1), stride=(1, 1, 1), bias=True),  nn.ELU(),)

        self.inception_out_feature_size = 64

        self.spatial_temporal_net_1n = nn.ModuleList([ nn.Sequential(
            nn.BatchNorm3d(increase_dimension_hidden_size),
            nn.Conv3d(increase_dimension_hidden_size, self.inception_out_feature_size,
                      kernel_size=(1, args.top_k_neighbors , 1),
                      bias=True),
            nn.ELU()),
            nn.Sequential(
            nn.Conv3d(self.inception_out_feature_size, self.inception_out_feature_size,
                      kernel_size=(1, 1, args.seq_length_y,),
                      bias=True),
            nn.ELU())]
        )


        neighbors_list, seq_list = args.pooling_neighbors_list, args.pooling_seq_list
        # print("*****",args.max_allow_spatial_conv, args.seq_length_x, args.max_allow_spatial_conv, neighbors_list,seq_list)
        self.spatial_temporal_net_pooling = nn.ModuleList()
        for index, conv_parameters in enumerate(neighbors_list):
            if index == 0:
                norm_3d = nn.BatchNorm3d(increase_dimension_hidden_size)

                pool_net = nn.MaxPool3d(kernel_size=(1, neighbors_list[index], seq_list[index]),
                                        )
                conv3d_net = nn.Conv3d(increase_dimension_hidden_size, self.inception_out_feature_size,
                                       kernel_size=(1, 1, 1), padding_mode='replicate',
                                       bias=True)
            else:
                norm_3d = nn.BatchNorm3d(self.inception_out_feature_size)
                pool_net = nn.MaxPool3d(kernel_size=(1, neighbors_list[index], seq_list[index]),
                                        )
                conv3d_net = nn.Conv3d(self.inception_out_feature_size, self.inception_out_feature_size,
                                       kernel_size=(1, 1, 1), padding_mode='replicate',
                                       bias=True)

            concat_net = nn.Sequential(norm_3d, pool_net, conv3d_net,  nn.ELU(),)
            print("spatial_temporal_net_pooling", pool_net)
            self.spatial_temporal_net_pooling.add_module('edge_att_gcn_activations_{}'.format(index), concat_net)

        Parameter_Model = Caculate_parameters_conv(args.top_k_neighbors, args.seq_length_x, args.max_allow_spatial_conv,
                                                   args.max_allow_dilation, weight = 'std')
        conv_bags = Parameter_Model.main()
        print("conv_bags = ", conv_bags)
        self.spatial_temporal_net_nn = nn.ModuleList()
        for index, conv_parameters in enumerate(conv_bags):
            if index == 0:
                norm_3d = nn.BatchNorm3d(increase_dimension_hidden_size)
                conv3d_net = nn.Conv3d(increase_dimension_hidden_size, self.inception_out_feature_size,
                                       kernel_size=(1, conv_parameters[0], conv_parameters[1]),
                                       dilation=(1, conv_parameters[2], conv_parameters[3]), padding_mode='replicate',
                                       bias=True)
            else:
                norm_3d = nn.BatchNorm3d(self.inception_out_feature_size)
                conv3d_net = nn.Conv3d(self.inception_out_feature_size, self.inception_out_feature_size,
                                       kernel_size=(1, conv_parameters[0], conv_parameters[1]),
                                       dilation=(1, conv_parameters[2], conv_parameters[3]), padding_mode='replicate',
                                       bias=True)
            concat_net = nn.Sequential(norm_3d, conv3d_net,  nn.ELU(),)
            print(" spatial_temporal_net_nn ",conv3d_net)
            self.spatial_temporal_net_nn.add_module('edge_att_gcn_activations_{}'.format(index), concat_net)

        self.final_conv= nn.Sequential(
                nn.Conv3d(self.inception_out_feature_size*3, 64, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                          bias=True),
                nn.BatchNorm3d(64),
                nn.ELU(),
                nn.Conv3d(64, 12, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                          bias=True),
            )

    def forward(self, threeDinput):
        # input shape (batch_size, number_sensors, neighbors, 1, in-seq)
        threeDinput = threeDinput.permute(0, 3, 1, 2, 4).contiguous()
        # print("  shape of threeDinput is ", threeDinput.shape)
        threeDinput = self.increase_dimension_net(threeDinput)
        # print(" shaoe of spatial_temporal_net ", threeDinput.shape)

        output_nn = threeDinput.clone()
        output_pooling = threeDinput.clone()
        # print(" shaoe of spatial_temporal_net ", output_pooling.shape)
        for index, net in enumerate(self.spatial_temporal_net_nn):
            output_net = net(output_nn)
            size_neighbor, size_temporal = output_net.shape[3],output_net.shape[4]
            # print(" shaoe of spatial_temporal_net ",output_net.shape, output.shape )
            if index > 0 :
                # print(" shaoe of spatial_temporal_net ", net, size_neighbor, size_temporal,   output[:,:,:,-size_neighbor:,-size_temporal:].shape, output_net.shape)
                output_nn = output_nn[:,:,:,-size_neighbor:,-size_temporal:] + output_net
            else:
                output_nn = output_net
        # print(" shaoe of spatial_temporal_net ",output_pooling.shape )
        for index, net in enumerate(self.spatial_temporal_net_pooling):
            output_net = net(output_pooling)
            size_neighbor, size_temporal = output_net.shape[3], output_net.shape[4]
            # print(" shaoe of spatial_temporal_net ",output_net.shape, output.shape )
            if index > 0:
                # print(" shaoe of spatial_temporal_net ",  output[:,:,:,-size_neighbor:,-size_temporal:] .shape, output_net.shape)
                output_pooling = output_pooling[:, :, :, -size_neighbor:, -size_temporal:] + output_net
            else:
                output_pooling = output_net

        for index, net in enumerate(self.spatial_temporal_net_1n):
            output_1n = net(threeDinput)
            size_neighbor, size_temporal = output_1n.shape[3], output_net.shape[4]
            # print(" shaoe of spatial_temporal_net ",output_net.shape, output.shape )
            if index > 0:
                # print(" shaoe of spatial_temporal_net ", size_neighbor,size_temporal)
                # print(threeDinput[:,:,:,-size_neighbor:,-size_temporal:].shape, output_1n.shape)
                threeDinput = threeDinput[:, :, :, -size_neighbor:, -size_temporal:] + output_1n
            else:
                threeDinput = output_1n


        output = torch.cat([output_pooling, output_nn, output_1n ], dim = 1)


        output = self.final_conv(output)
        output = output.squeeze(-1)
        return output

class Spatial_temporal_Inception_Pure_v3(nn.Module):
    #### Residual connections based on 1*1*1+ continuous hidden size of features,
    def __init__(self, args):
        super(Spatial_temporal_Inception_Pure_v3, self).__init__()
        args.spatial_temporal_hidden.append(args.seq_length_y)
        self.increase_dimension_hidden_size = 32
        self.increase_dimension_net = nn.Sequential(nn.Conv3d(1, self.increase_dimension_hidden_size, kernel_size=(1, 1, 1),
                                            dilation=(1, 1, 1), stride=(1, 1, 1), bias=True),  nn.ELU(),)

        self.inception_out_feature_size = 64
        hidden_for_1n = [self.increase_dimension_hidden_size, int((self.increase_dimension_hidden_size + self.inception_out_feature_size)/2), self.inception_out_feature_size ]

        self.spatial_temporal_net_1n = nn.ModuleList([nn.Sequential(
            nn.BatchNorm3d(hidden_for_1n[0]),
            nn.Conv3d(hidden_for_1n[0], hidden_for_1n[1],
                      kernel_size=(1, args.top_k_neighbors, 1),
                      bias=True),
            nn.ELU(), ),
            nn.Sequential(
                nn.Conv3d(hidden_for_1n[1], hidden_for_1n[2],
                          kernel_size=(1, 1, args.seq_length_y),
                          bias=True),
                nn.ELU(), ), ]
        )

        self.spatial_temporal_net_1n_assist = nn.ModuleList([
            nn.Conv3d(hidden_for_1n[0], hidden_for_1n[1],
                      kernel_size=(1, 1, 1),
                      bias=True),
            nn.Conv3d(hidden_for_1n[1], hidden_for_1n[2],
                      kernel_size=(1, 1, 1),
                      bias=True), ]
        )

        neighbors_list, seq_list = args.pooling_neighbors_list, args.pooling_seq_list
        # print("*****",args.max_allow_spatial_conv, args.seq_length_x, args.max_allow_spatial_conv, neighbors_list,seq_list)
        hidden_for_pooling = [self.increase_dimension_hidden_size]
        for i in range(len(neighbors_list)):
            cur = self.increase_dimension_hidden_size + (self.inception_out_feature_size - self.increase_dimension_hidden_size) /len(neighbors_list) *(i+1)
            hidden_for_pooling.append(int(cur))
        self.spatial_temporal_net_pooling_assist = nn.ModuleList()
        self.spatial_temporal_net_pooling = nn.ModuleList()
        for index, conv_parameters in enumerate(neighbors_list):
            norm_3d = nn.BatchNorm3d(hidden_for_pooling[index])

            pool_net = nn.MaxPool3d(kernel_size=(1, neighbors_list[index], seq_list[index]),
                                    )
            conv3d_net = nn.Conv3d(hidden_for_pooling[index],hidden_for_pooling[index+1],
                                   kernel_size=(1, 1, 1), padding_mode='replicate',
                                   bias=True)

            concat_net = nn.Sequential(norm_3d, pool_net, conv3d_net,  nn.ELU(),)
            print("spatial_temporal_net_pooling", pool_net)
            self.spatial_temporal_net_pooling.add_module('spatial_temporal_net_pooling{}'.format(index), concat_net)

            conv3d_net_assist = nn.Conv3d(hidden_for_pooling[index], hidden_for_pooling[index + 1],
                                   kernel_size=(1, 1, 1), padding_mode='replicate',
                                   bias=True)
            self.spatial_temporal_net_pooling_assist.add_module('spatial_temporal_net_pooling_assist_{}'.format(index),
                                                                conv3d_net_assist)






        Parameter_Model = Caculate_parameters_conv(args.top_k_neighbors, args.seq_length_x, args.max_allow_spatial_conv,
                                                   args.max_allow_dilation, weight = 'std')
        conv_bags = Parameter_Model.main()
        print("conv_bags = ", conv_bags)
        hidden_for_nn = [self.increase_dimension_hidden_size]
        for i in range(len(conv_bags)):
            cur = self.increase_dimension_hidden_size + (
                        self.inception_out_feature_size - self.increase_dimension_hidden_size) / len(conv_bags) * (
                              i + 1)
            hidden_for_nn.append(int(cur))

        self.spatial_temporal_net_nn_assist = nn.ModuleList()

        self.spatial_temporal_net_nn = nn.ModuleList()
        for index, conv_parameters in enumerate(conv_bags):

            norm_3d = nn.BatchNorm3d(hidden_for_nn[index])
            conv3d_net = nn.Conv3d(hidden_for_nn[index], hidden_for_nn[index+1],
                                   kernel_size=(1, conv_parameters[0], conv_parameters[1]),
                                   dilation=(1, conv_parameters[2], conv_parameters[3]), padding_mode='replicate',
                                   bias=True)

            concat_net = nn.Sequential(norm_3d, conv3d_net,  nn.ELU(),)
            print(" spatial_temporal_net_nn ",conv3d_net)
            self.spatial_temporal_net_nn.add_module('spatial_temporal_net_nn{}'.format(index), concat_net)
            conv3d_net_assist = nn.Conv3d(hidden_for_nn[index], hidden_for_nn[index + 1],
                                   kernel_size=(1, 1, 1), bias=True)
            self.spatial_temporal_net_nn_assist.add_module('spatial_temporal_net_nn_assist{}'.format(index), conv3d_net_assist)

        self.final_conv= nn.Sequential(
                nn.Conv3d(self.inception_out_feature_size*3, 64, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                          bias=True),
                nn.BatchNorm3d(64),
                nn.ELU(),
                nn.Conv3d(64, 12, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                          bias=True),
            )

    def forward(self, threeDinput):
        # input shape (batch_size, number_sensors, neighbors, 1, in-seq)
        threeDinput = threeDinput.permute(0, 3, 1, 2, 4).contiguous()
        # print("  shape of threeDinput is ", threeDinput.shape)
        threeDinput = self.increase_dimension_net(threeDinput)
        # print(" shaoe of spatial_temporal_net ", threeDinput.shape)

        output_nn = threeDinput.clone()
        output_pooling = threeDinput.clone()
        output_1n = threeDinput.clone()
        # print(" shaoe of spatial_temporal_net ", output_pooling.shape)
        for index, (net, net_assist) in enumerate(zip(self.spatial_temporal_net_nn,self.spatial_temporal_net_nn_assist)):
            output_net = net(output_nn)
            size_neighbor, size_temporal = output_net.shape[3],output_net.shape[4]
            output_nn = net_assist(output_nn)[:,:,:,-size_neighbor:,-size_temporal:] + output_net

        # print(" shaoe of spatial_temporal_net ",output_pooling.shape )
        for index,(net, net_assist) in enumerate(zip(self.spatial_temporal_net_pooling,self.spatial_temporal_net_pooling_assist)):
            output_net = net(output_pooling)
            size_neighbor, size_temporal = output_net.shape[3], output_net.shape[4]
            output_pooling = net_assist(output_pooling)[:, :, :, -size_neighbor:, -size_temporal:] + output_net


        for index, (net, net_assist) in enumerate(zip(self.spatial_temporal_net_1n,self.spatial_temporal_net_1n_assist)):
            output_net = net(output_1n)
            size_neighbor, size_temporal = output_net.shape[3], output_net.shape[4]
            # print(" shaoe of spatial_temporal_net ",output_net.shape, output.shape )
            output_1n = net_assist(output_1n)[:, :, :, -size_neighbor:, -size_temporal:] + output_net

        output = torch.cat([output_pooling, output_nn, output_1n ], dim = 1)

        output = self.final_conv(output)
        output = output.squeeze(-1)
        return output

class Spatial_temporal_Inception_Pure_v4(nn.Module):
    #### Residual connections + sharp hidden size of features,
    def __init__(self, args):
        super(Spatial_temporal_Inception_Pure_v4, self).__init__()
        args.spatial_temporal_hidden.append(args.seq_length_y)
        increase_dimension_hidden_size = 32
        self.increase_dimension_net = nn.Sequential(nn.Conv3d(1, increase_dimension_hidden_size, kernel_size=(1, 1, 1),
                                            dilation=(1, 1, 1), stride=(1, 1, 1), bias=True),  nn.ELU(),)

        self.inception_out_feature_size = 64

        increase_dimension_hidden_size = increase_dimension_hidden_size + 1
        self.spatial_temporal_net_1n = nn.ModuleList([ nn.Sequential(
            nn.BatchNorm3d(increase_dimension_hidden_size),
            nn.Conv3d(increase_dimension_hidden_size, self.inception_out_feature_size,
                      kernel_size=(1, args.top_k_neighbors , 1),
                      bias=True),
            nn.ELU()),
            nn.Sequential(
            nn.Conv3d(self.inception_out_feature_size, self.inception_out_feature_size,
                      kernel_size=(1, 1, args.seq_length_y,),
                      bias=True),
            nn.ELU())]
        )


        neighbors_list, seq_list = args.pooling_neighbors_list, args.pooling_seq_list
        # print("*****",args.max_allow_spatial_conv, args.seq_length_x, args.max_allow_spatial_conv, neighbors_list,seq_list)
        self.spatial_temporal_net_pooling = nn.ModuleList()
        for index, conv_parameters in enumerate(neighbors_list):
            if index == 0:
                norm_3d = nn.BatchNorm3d(increase_dimension_hidden_size)

                pool_net = nn.MaxPool3d(kernel_size=(1, neighbors_list[index], seq_list[index]),
                                        )
                conv3d_net = nn.Conv3d(increase_dimension_hidden_size, self.inception_out_feature_size,
                                       kernel_size=(1, 1, 1), padding_mode='replicate',
                                       bias=True)
            else:
                norm_3d = nn.BatchNorm3d(self.inception_out_feature_size)
                pool_net = nn.MaxPool3d(kernel_size=(1, neighbors_list[index], seq_list[index]),
                                        )
                conv3d_net = nn.Conv3d(self.inception_out_feature_size, self.inception_out_feature_size,
                                       kernel_size=(1, 1, 1), padding_mode='replicate',
                                       bias=True)

            concat_net = nn.Sequential(norm_3d, pool_net, conv3d_net,  nn.ELU(),)
            print("spatial_temporal_net_pooling", pool_net)
            self.spatial_temporal_net_pooling.add_module('edge_att_gcn_activations_{}'.format(index), concat_net)

        Parameter_Model = Caculate_parameters_conv(args.top_k_neighbors, args.seq_length_x, args.max_allow_spatial_conv,
                                                   args.max_allow_dilation, weight = 'std')
        conv_bags = Parameter_Model.main()
        print("conv_bags = ", conv_bags)
        self.spatial_temporal_net_nn = nn.ModuleList()
        for index, conv_parameters in enumerate(conv_bags):
            if index == 0:
                norm_3d = nn.BatchNorm3d(increase_dimension_hidden_size)
                conv3d_net = nn.Conv3d(increase_dimension_hidden_size, self.inception_out_feature_size,
                                       kernel_size=(1, conv_parameters[0], conv_parameters[1]),
                                       dilation=(1, conv_parameters[2], conv_parameters[3]), padding_mode='replicate',
                                       bias=True)
            else:
                norm_3d = nn.BatchNorm3d(self.inception_out_feature_size)
                conv3d_net = nn.Conv3d(self.inception_out_feature_size, self.inception_out_feature_size,
                                       kernel_size=(1, conv_parameters[0], conv_parameters[1]),
                                       dilation=(1, conv_parameters[2], conv_parameters[3]), padding_mode='replicate',
                                       bias=True)
            concat_net = nn.Sequential(norm_3d, conv3d_net,  nn.ELU(),)
            print(" spatial_temporal_net_nn ",conv3d_net)
            self.spatial_temporal_net_nn.add_module('edge_att_gcn_activations_{}'.format(index), concat_net)

        self.final_conv= nn.Sequential(
                nn.Conv3d(self.inception_out_feature_size*3, 64, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                          bias=True),
                nn.BatchNorm3d(64),
                nn.ELU(),
                nn.Conv3d(64, 12, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                          bias=True),
            )

    def forward(self, threeDinput):
        # input shape (batch_size, number_sensors, neighbors, 1, in-seq)
        threeDinput = threeDinput.permute(0, 3, 1, 2, 4).contiguous()
        # print("  shape of threeDinput is ", threeDinput.shape)

        threeDinput_increase = self.increase_dimension_net(threeDinput)
        threeDinput = torch.cat([threeDinput_increase, threeDinput], dim=1)
        # print(" shaoe of spatial_temporal_net ", threeDinput.shape)

        output_nn = threeDinput.clone()
        output_pooling = threeDinput.clone()
        output_1n = threeDinput.clone()
        # print(" shaoe of spatial_temporal_net ", output_pooling.shape)
        # for index, net in enumerate(self.spatial_temporal_net_nn):
        #     output_net = net(output_nn)
        #     size_neighbor, size_temporal = output_net.shape[3], output_net.shape[4]
        #     # print(" shaoe of spatial_temporal_net ",output_net.shape, output.shape )
        #     if index > 0:
        #         # print(" shaoe of spatial_temporal_net ", net, size_neighbor, size_temporal,   output[:,:,:,-size_neighbor:,-size_temporal:].shape, output_net.shape)
        #         output_nn = output_nn[:, :, :, -size_neighbor:, -size_temporal:] + output_net
        #     else:
        #         output_nn = output_net

        for index, net in enumerate(self.spatial_temporal_net_nn):
            output_net = net(output_nn)
            size_neighbor, size_temporal = output_net.shape[3],output_net.shape[4]
            # print(" shaoe of spatial_temporal_net ",output_net.shape, output.shape )
            if index > 0 :
                # print(" shaoe of spatial_temporal_net ", net, size_neighbor, size_temporal,   output[:,:,:,-size_neighbor:,-size_temporal:].shape, output_net.shape)
                output_nn  = output_nn[:,:,:,-size_neighbor:,-size_temporal:] + output_net

            else:
                feature_dim, total_dim = output_nn.shape[1], output_net.shape[1]
                temp = output_nn[:, :, :, -size_neighbor:,
                                                        -size_temporal:] + output_net[:, -feature_dim:, :, :, :]
                output_nn = torch.cat([output_net[:, :(total_dim - feature_dim), :, :, :], temp], axis = 1)



                # print(" shaoe of spatial_temporal_net ",output_pooling.shape )
        for index, net in enumerate(self.spatial_temporal_net_pooling):
            output_net = net(output_pooling)
            size_neighbor, size_temporal = output_net.shape[3], output_net.shape[4]
            # print(" shaoe of spatial_temporal_net ",output_net.shape, output.shape )
            if index > 0:
                # print(" shaoe of spatial_temporal_net ",  output[:,:,:,-size_neighbor:,-size_temporal:] .shape, output_net.shape)
                output_pooling = output_pooling[:, :, :, -size_neighbor:, -size_temporal:] + output_net
            else:
                feature_dim, total_dim = output_pooling.shape[1], output_net.shape[1]
                temp  = output_pooling[:, :, :, -size_neighbor:,
                                                        -size_temporal:] + output_net[:, -feature_dim:, :, :, :]
                output_pooling = torch.cat([output_net[:, :(total_dim - feature_dim), :, :, :], temp], axis=1)

        for index, net in enumerate(self.spatial_temporal_net_1n):
            output_net = net(output_1n)
            size_neighbor, size_temporal = output_net.shape[3], output_net.shape[4]
            # print(" shaoe of spatial_temporal_net ",output_net.shape, output.shape )
            if index > 0:
                # print(" shaoe of spatial_temporal_net ", size_neighbor,size_temporal)
                # print(threeDinput[:,:,:,-size_neighbor:,-size_temporal:].shape, output_1n.shape)
                output_1n = output_1n[:, :, :, -size_neighbor:, -size_temporal:] + output_net
            else:
                feature_dim, total_dim = output_1n.shape[1], output_net.shape[1]
                temp  = output_1n[:, :, :, -size_neighbor:,
                                                        -size_temporal:] + output_net[:, -feature_dim:, :, :, :]
                output_1n = torch.cat([output_net[:, :(total_dim - feature_dim), :, :, :], temp], axis=1)

        output = torch.cat([output_pooling, output_nn, output_1n ], dim = 1)


        output = self.final_conv(output)
        output = output.squeeze(-1)
        return output

class Spatial_temporal_Inception_Pure_v5(nn.Module):
    #### Residual connections + sharp hidden size of features,
    def __init__(self, args):
        super(Spatial_temporal_Inception_Pure_v5, self).__init__()
        args.spatial_temporal_hidden.append(args.seq_length_y)
        increase_dimension_hidden_size = 32
        self.increase_dimension_net = nn.Sequential(nn.Conv3d(1, increase_dimension_hidden_size, kernel_size=(1, 1, 1),
                                            dilation=(1, 1, 1), stride=(1, 1, 1), bias=True),  nn.ELU(),)

        self.inception_out_feature_size = 64

        increase_dimension_hidden_size = increase_dimension_hidden_size + 1
        self.spatial_temporal_net_1n = nn.ModuleList([ nn.Sequential(
            nn.BatchNorm3d(increase_dimension_hidden_size),
            nn.Conv3d(increase_dimension_hidden_size, self.inception_out_feature_size,
                      kernel_size=(1, args.top_k_neighbors , 1),
                      bias=True),
            nn.ELU()),
            nn.Sequential(
            nn.Conv3d(self.inception_out_feature_size, self.inception_out_feature_size,
                      kernel_size=(1, 1, args.seq_length_y,),
                      bias=True),
            nn.ELU())]
        )


        neighbors_list, seq_list = args.pooling_neighbors_list, args.pooling_seq_list
        # print("*****",args.max_allow_spatial_conv, args.seq_length_x, args.max_allow_spatial_conv, neighbors_list,seq_list)
        self.spatial_temporal_net_pooling = nn.ModuleList()
        for index, conv_parameters in enumerate(neighbors_list):
            if index == 0:
                norm_3d = nn.BatchNorm3d(increase_dimension_hidden_size)

                pool_net = nn.MaxPool3d(kernel_size=(1, neighbors_list[index], seq_list[index]),
                                        )
                conv3d_net = nn.Conv3d(increase_dimension_hidden_size, self.inception_out_feature_size,
                                       kernel_size=(1, 1, 1), padding_mode='replicate',
                                       bias=True)
            else:
                norm_3d = nn.BatchNorm3d(self.inception_out_feature_size)
                pool_net = nn.MaxPool3d(kernel_size=(1, neighbors_list[index], seq_list[index]),
                                        )
                conv3d_net = nn.Conv3d(self.inception_out_feature_size, self.inception_out_feature_size,
                                       kernel_size=(1, 1, 1), padding_mode='replicate',
                                       bias=True)

            concat_net = nn.Sequential(norm_3d, pool_net, conv3d_net,  nn.ELU(),)
            print("spatial_temporal_net_pooling", pool_net)
            self.spatial_temporal_net_pooling.add_module('edge_att_gcn_activations_{}'.format(index), concat_net)

        Parameter_Model = Caculate_parameters_conv(args.top_k_neighbors, args.seq_length_x, args.max_allow_spatial_conv,
                                                   args.max_allow_dilation, weight = 'std')
        conv_bags = Parameter_Model.main()
        print("conv_bags = ", conv_bags)
        self.spatial_temporal_net_nn = nn.ModuleList()
        for index, conv_parameters in enumerate(conv_bags):
            if index == 0:
                norm_3d = nn.BatchNorm3d(increase_dimension_hidden_size)
                conv3d_net = nn.Conv3d(increase_dimension_hidden_size, self.inception_out_feature_size,
                                       kernel_size=(1, conv_parameters[0], conv_parameters[1]),
                                       dilation=(1, conv_parameters[2], conv_parameters[3]), padding_mode='replicate',
                                       bias=True)
            else:
                norm_3d = nn.BatchNorm3d(self.inception_out_feature_size)
                conv3d_net = nn.Conv3d(self.inception_out_feature_size, self.inception_out_feature_size,
                                       kernel_size=(1, conv_parameters[0], conv_parameters[1]),
                                       dilation=(1, conv_parameters[2], conv_parameters[3]), padding_mode='replicate',
                                       bias=True)
            concat_net = nn.Sequential(norm_3d, conv3d_net,  nn.ELU(),)
            print(" spatial_temporal_net_nn ",conv3d_net)
            self.spatial_temporal_net_nn.add_module('edge_att_gcn_activations_{}'.format(index), concat_net)

        self.spatial_temporal_net_lq = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(1, args.top_k_neighbors, args.seq_length_x),
                      bias=True),
            nn.ELU(), )

        self.final_conv= nn.Sequential(
                nn.Conv3d(self.inception_out_feature_size*4, 64, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                          bias=True),
                nn.BatchNorm3d(64),
                nn.ELU(),
                nn.Conv3d(64, 12, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                          bias=True),
            )

    def forward(self, threeDinput):
        # input shape (batch_size, number_sensors, neighbors, 1, in-seq)
        threeDinput = threeDinput.permute(0, 3, 1, 2, 4).contiguous()
        # print("  shape of threeDinput is ", threeDinput.shape)
        input_for_lq = threeDinput.clone()
        threeDinput_increase = self.increase_dimension_net(threeDinput)
        threeDinput = torch.cat([threeDinput_increase, threeDinput], dim=1)
        # print(" shaoe of spatial_temporal_net ", threeDinput.shape)

        output_nn = threeDinput.clone()
        output_pooling = threeDinput.clone()
        output_1n = threeDinput.clone()


        for index, net in enumerate(self.spatial_temporal_net_nn):
            output_net = net(output_nn)
            size_neighbor, size_temporal = output_net.shape[3],output_net.shape[4]
            # print(" shaoe of spatial_temporal_net ",output_net.shape, output.shape )
            if index > 0 :
                # print(" shaoe of spatial_temporal_net ", net, size_neighbor, size_temporal,   output[:,:,:,-size_neighbor:,-size_temporal:].shape, output_net.shape)
                output_nn  = output_nn[:,:,:,-size_neighbor:,-size_temporal:] + output_net

            else:
                feature_dim, total_dim = output_nn.shape[1], output_net.shape[1]
                temp = output_nn[:, :, :, -size_neighbor:,
                                                        -size_temporal:] + output_net[:, -feature_dim:, :, :, :]
                output_nn = torch.cat([output_net[:, :(total_dim - feature_dim), :, :, :], temp], axis = 1)



                # print(" shaoe of spatial_temporal_net ",output_pooling.shape )
        for index, net in enumerate(self.spatial_temporal_net_pooling):
            output_net = net(output_pooling)
            size_neighbor, size_temporal = output_net.shape[3], output_net.shape[4]
            # print(" shaoe of spatial_temporal_net ",output_net.shape, output.shape )
            if index > 0:
                # print(" shaoe of spatial_temporal_net ",  output[:,:,:,-size_neighbor:,-size_temporal:] .shape, output_net.shape)
                output_pooling = output_pooling[:, :, :, -size_neighbor:, -size_temporal:] + output_net
            else:
                feature_dim, total_dim = output_pooling.shape[1], output_net.shape[1]
                temp  = output_pooling[:, :, :, -size_neighbor:,
                                                        -size_temporal:] + output_net[:, -feature_dim:, :, :, :]
                output_pooling = torch.cat([output_net[:, :(total_dim - feature_dim), :, :, :], temp], axis=1)

        for index, net in enumerate(self.spatial_temporal_net_1n):
            output_net = net(output_1n)
            size_neighbor, size_temporal = output_net.shape[3], output_net.shape[4]
            # print(" shaoe of spatial_temporal_net ",output_net.shape, output.shape )
            if index > 0:
                # print(" shaoe of spatial_temporal_net ", size_neighbor,size_temporal)
                # print(threeDinput[:,:,:,-size_neighbor:,-size_temporal:].shape, output_1n.shape)
                output_1n = output_1n[:, :, :, -size_neighbor:, -size_temporal:] + output_net
            else:
                feature_dim, total_dim = output_1n.shape[1], output_net.shape[1]
                temp  = output_1n[:, :, :, -size_neighbor:,
                                                        -size_temporal:] + output_net[:, -feature_dim:, :, :, :]
                output_1n = torch.cat([output_net[:, :(total_dim - feature_dim), :, :, :], temp], axis=1)

        output_lq = self.spatial_temporal_net_lq(input_for_lq)

        output = torch.cat([output_pooling, output_nn, output_1n, output_lq ], dim = 1)


        output = self.final_conv(output)
        output = output.squeeze(-1)
        return output



class EmbedGCN(nn.Module):

    def __init__(self, args, connections_road, edge_attribute, sensor_indexes, writer =None):
        super(EmbedGCN, self).__init__()
        self.args = args
        self.top_k_values = args.top_k_neighbors
        self.writer = writer
        self.static_feature_module = Static_features(args, connections_road, edge_attribute, sensor_indexes)
        self.dynamic_feature_module = Dynamic_features(args)
        if args.Version_id == 'v1':
            self.spatial_temporal = Spatial_temporal_Inception_Pure_v1(args)
        elif args.Version_id == 'v2':
            self.spatial_temporal = Spatial_temporal_Inception_Pure_v2(args)
        elif args.Version_id == 'v3':
            self.spatial_temporal = Spatial_temporal_Inception_Pure_v3(args)
        elif args.Version_id == 'v4':
            self.spatial_temporal = Spatial_temporal_Inception_Pure_v4(args)
        elif args.Version_id == 'v5':
            self.spatial_temporal = Spatial_temporal_Inception_Pure_v5(args)
        # elif args.Version_id == 'v6':
        #     self.spatial_temporal = Spatial_temporal_Inception_Pure_v6(args)

        self.global_terater = 0

    def cosine_simularity(self,x):
        ## shape of x is (batch_size,  # sensors,fused_dim)
        x = x.permute((1, 2, 0))
        cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
        cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
        return cos_sim_pairwise

    def vis_tensor(self, tensors, print_every=100):
        if self.writer and self.global_terater % print_every == 0:
            self.writer.add_image('cor_mat_realtime',
                                       tensors,
                                       global_step= self.global_terater,
                                       dataformats='HW')
    def vis_time_series(self, timeseries, sensores, print_every=100,):
        if self.writer and self.global_terater % print_every == 0:
            timeseries = timeseries.cpu().numpy()
            import matplotlib.pyplot as plt
            plt.switch_backend('agg')
            fig = plt.figure()
            plt.legend(sensores.cpu().numpy())
            plt.plot(list(range(timeseries.shape[-1])), timeseries.transpose())

            self.writer.add_figure('time_series_data',
                                       fig,
                                       global_step=self.global_terater,
                                       )

    def forward(self, input):
        self.global_terater =   self.global_terater + 1
        # input shape (batch_size, in_dim, #edges, in_seq)
        # output shape (batch_size, in_seq, #edges, in_dim)
        # print("input  = ", input.shape,input[2,:, 122:125,:5] )

        input_for_dynamic_features = input.clone().detach()
        originalIinput = input.clone().detach()
        dynamic_fea = self.dynamic_feature_module(input_for_dynamic_features)
        static_fea = self.static_feature_module()
        number_sensors, feature_size = static_fea.shape[0], static_fea.shape[1]
        batch_size = dynamic_fea.shape[0]
        # print(" static_fea --11", static_fea.shape, static_fea[0:5,:])
        static_fea_expand = static_fea.unsqueeze(-1).expand(number_sensors, feature_size, batch_size)
        # print(" static_fea --22", static_fea_expand.shape,static_fea_expand[0:5,:,0:2])
        static_fea_expand = static_fea_expand.permute(2, 0, 1)  # shape is (batch_size,#sensors, dim)
        # print(" static_fea --33", static_fea_expand.shape,static_fea_expand[0:2,0:5,:])


        if self.args.fusion == 'dynamic':
            fushed_features = dynamic_fea
        elif  self.args.fusion == 'static':
            fushed_features = static_fea_expand
        elif  self.args.fusion == 'concat':
            fushed_features = torch.cat([static_fea_expand, dynamic_fea],
                                        axis=2)  # shape is (batch_size,#sensors,fused_dim)
        else:
            assert 1==0,  f"fusion mechanism {self.args.fusion} is not defined!!"




        adjacency_mat = self.cosine_simularity(fushed_features)  # shape is (batch_size, #sensors, #sensors)
        # print(" adjacency_mat from cosine = ", adjacency_mat[0,:10,:10])
        # self.vis_tensor(adjacency_mat[0,:,:])

        adjacency_mat_topk_values, adjacency_mat_topk_indexes = adjacency_mat.topk(self.top_k_values, dim=2,
                                                                                   largest=True, sorted=True)
        # print(" top k index = ", adjacency_mat_topk_values[0,:10,:10])

        number_sensors, seq_in = input.shape[2], input.shape[3]

        input = input.permute(0,  1,2, 3).expand(-1,  number_sensors,-1,
                                                 -1)  # input shape (batch_size, #edges, #edges,in_seq)

        # print("input and index  = ", input.shape, adjacency_mat_topk_indexes.shape,adjacency_mat_topk_indexes[:,:1,:])
        adjacency_mat_topk_indexes = adjacency_mat_topk_indexes.unsqueeze(-1).expand(-1, -1, -1, seq_in)
        threeDinput = torch.gather(input, 2, adjacency_mat_topk_indexes)

        adjacency_mat_topk_values = adjacency_mat_topk_values.unsqueeze(-1).expand(-1, -1, -1, seq_in)
        threeDinput = threeDinput * adjacency_mat_topk_values

        threeDinput = threeDinput.unsqueeze(3).contiguous() # shape of threeDinput is (batch_size, number_sensors, neighbors, 1, in-seq)
        if False:
            print("input and index 22  = ", threeDinput.shape, adjacency_mat_topk_indexes.shape )
            print(" index =  ", adjacency_mat_topk_indexes[0,101:102,:,0], adjacency_mat_topk_indexes[0,101:102,0,0].squeeze())
            print(" one way: ", threeDinput[0,101:102,:,:,:] )
            for i in range(6):
                print(" another way: ", originalIinput.shape, adjacency_mat_topk_indexes[0,101:102,i,0],  originalIinput[0,:, adjacency_mat_topk_indexes[0,101:102,i,0], :])
                print(" after transpose  way: ",              input[0, 101:102, adjacency_mat_topk_indexes[0, 101:102, i, 0], :])

        threeDinput_out = threeDinput.clone().detach()
        threeDinput_out = threeDinput_out[0,0,:,:,:].squeeze()
        # print("  threeDinput_out  = ",threeDinput_out.shape)
        # self.vis_time_series(threeDinput_out, adjacency_mat_topk_indexes[0,0, :,0])



        output = self.spatial_temporal(threeDinput)
        # output = output.view(batch_size, number_sensors, -1, seq_in)
        # output = output.permute(0, 3, 1, 2)

        return output, self.cosine_simularity(static_fea_expand)

