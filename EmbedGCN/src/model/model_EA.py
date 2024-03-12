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
import pickle
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


class ResNet_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, input_channel = 1 , output_channel = 256, resnet_layers = [1, 2, 2, 2], neighbors = 12 ):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, resnet_layers, neighbors)

    def forward(self, input):
        return self.ConvNet(input)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv1x3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = self._conv1x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv1x3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv3d(in_planes, out_planes, kernel_size=(1,3,3), stride=(1,stride,stride),
                         padding=(0,1,1), bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # print('in residual connections ',out.shape, x.shape, residual.shape)
        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers, neighbors):
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv3d(input_channel, int(output_channel / 16),
                                 kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)
        self.bn0_1 = nn.BatchNorm3d(int(output_channel / 16))
        self.conv0_2 = nn.Conv3d(int(output_channel / 16), self.inplanes,
                                 kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)
        self.bn0_2 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv3d(self.output_channel_block[0], self.output_channel_block[
                               0], kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)
        self.bn1 = nn.BatchNorm3d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv3d(self.output_channel_block[1], self.output_channel_block[
                               1], kernel_size=(1,2,2), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(self.output_channel_block[1])

        if neighbors in [8,10]:
            self.maxpool3 = nn.MaxPool3d(kernel_size=(1,1,2), stride=(1,1, 2), padding=(0,0, 1))
            self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
            self.conv3 = nn.Conv3d(self.output_channel_block[2], self.output_channel_block[
                2], kernel_size=(1, 1, 2), stride=1, padding=(0, 0, 0), bias=False)
            self.bn3 = nn.BatchNorm3d(self.output_channel_block[2])
        elif neighbors in [12, 14, 16]:
            self.maxpool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
            self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
            self.conv3 = nn.Conv3d(self.output_channel_block[2], self.output_channel_block[
                2], kernel_size=(1, 2, 2), stride=1, padding=(0, 0, 0), bias=False)
            self.bn3 = nn.BatchNorm3d(self.output_channel_block[2])
        # elif neighbors in [18]:
        #     self.maxpool3 = nn.Sequential(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1)),
        #                                   self._make_layer(block, self.output_channel_block[2], 1,
        #                                                    stride=1),
        #                                   )
        #     self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        #     self.conv3 = nn.Conv3d(self.output_channel_block[2], self.output_channel_block[
        #         2], kernel_size=(1, 2, 2), stride=1, padding=(0, 0, 0), bias=False)
        #     self.bn3 = nn.BatchNorm3d(self.output_channel_block[2])


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("resnet input ", x.shape)
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)
        # print("resnet  00", x.shape)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("maxpool2  before", x.shape)
        x = self.maxpool2(x)
        # print("maxpool2  after", x.shape)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)

        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # print("resnet 33 ", x.shape)
        # x = self.layer4(x)
        # x = self.conv4_1(x)
        # x = self.bn4_1(x)
        # x = self.relu(x)
        # x = self.conv4_2(x)
        # x = self.bn4_2(x)
        # x = self.relu(x)

        return x


class Spatial_temporal(nn.Module):

    def __init__(self, args):
        super(Spatial_temporal, self).__init__()
        # self.spatial_temporal_networks = nn.Sequential()
        channels_conv_start_layer = args.channels_conv_start_layer

        self.start_conv = self.final_convs= nn.Sequential(
            nn.Conv3d(1, channels_conv_start_layer, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                      bias=True),
            nn.ReLU(),
        )
        channels_conv_extractor_out =  args.channels_conv_extractor_out

        self.spatial_temporal_feature_extractor = ResNet_FeatureExtractor(input_channel = channels_conv_start_layer , output_channel = channels_conv_extractor_out,  resnet_layers = args.resnet_layers, neighbors = args.top_k_neighbors)


        self.final_convs= nn.Sequential(
            nn.Conv3d(channels_conv_extractor_out, int(channels_conv_extractor_out/2), kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                      bias=True),
            nn.ReLU(),
            nn.Conv3d(int(channels_conv_extractor_out/2), 12, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                      bias=True),
        )

    def forward(self, threeDinput):
        # input shape (batch_size, number_sensors, neighbors, 1, in-seq)
        threeDinput = threeDinput.permute(0,3,1,2,4).contiguous()
        # print("shape of input ",threeDinput.shape)

        output =  self.start_conv(threeDinput)
        # print("shape of start_conv", output.shape  )
        output = self.spatial_temporal_feature_extractor(output)
        # print("shape of spatial_temporal_feature_extractor", output.shape)

        output = self.final_convs(output)

        output = output.squeeze(-1)
        # output shape (batch_size, in_seq, #edges, in_dim)
        return output



class EmbedGCN(nn.Module):

    def __init__(self, args, connections_road, edge_attribute, sensor_indexes, writer =None):
        super(EmbedGCN, self).__init__()
        self.args = args
        self.top_k_values = args.top_k_neighbors
        self.writer = writer
        self.static_feature_module = Static_features(args, connections_road, edge_attribute, sensor_indexes)
        self.dynamic_feature_module = Dynamic_features(args)
        if args.Version_id == 'v_EA':
            self.spatial_temporal = Spatial_temporal(args)
        # elif args.Version_id == 'v2':
        #     self.spatial_temporal = Spatial_temporal_Inception_Pure_v2(args)
        # elif args.Version_id == 'v3':
        #     self.spatial_temporal = Spatial_temporal_Inception_Pure_v3(args)
        # elif args.Version_id == 'v4':
        #     self.spatial_temporal = Spatial_temporal_Inception_Pure_v4(args)
        # elif args.Version_id == 'v5':
        #     self.spatial_temporal = Spatial_temporal_Inception_Pure_v5(args)
        # elif args.Version_id == 'v6':
        #     self.spatial_temporal = Spatial_temporal_Inception_Pure_v6(args)

        self.cat_transform = nn.Linear(24,12)

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

    def output_tensor(self, tensors, print_every=100):
        if self.global_terater > 1:
            return
        tensors_out = tensors.cpu().numpy()
        print("shape of tensor is: ",tensors_out.shape)
        with open(f"output_tensor_la.pkl", 'wb') as fo:  # 将数据写入pkl文件
            pickle.dump(tensors_out, fo)


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
        # print("input  = ", input.shape,input[2,:, 122:125,:5])

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
        # self.output_tensor(adjacency_mat[:,:,:])

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

        # print(' threeDinput * adjacency_mat_topk_values =', threeDinput.shape, adjacency_mat_topk_values.shape)

        # threeDinput = threeDinput * adjacency_mat_topk_values
        # #
        threeDinput = threeDinput + adjacency_mat_topk_values

        # threeDinput = torch.cat([threeDinput, adjacency_mat_topk_values], dim =-1)
        # threeDinput = self.cat_transform(threeDinput)


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

