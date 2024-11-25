import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from Model_Base import *


#### Best for RUL
class FC_STGNN_RUL(nn.Module):
    def __init__(self, indim_fea, Conv_out, lstmhidden_dim, lstmout_dim, conv_kernel,hidden_dim, 
                time_length, num_node, num_windows, moving_window,stride,decay, pooling_choice, n_class):
        super(FC_STGNN_RUL, self).__init__()
        # graph_construction_type = args.graph_construction_type
        self.nonlin_map = Feature_extractor_1DCNN_RUL(1, lstmhidden_dim, lstmout_dim,kernel_size=conv_kernel)
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim*Conv_out, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim)
        )

        self.positional_encoding = PositionalEncoding(2*hidden_dim,0.1,max_len=5000)

        self.MPNN1 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, time_length, 
                                                time_window_size=moving_window[0], stride=stride[0], 
                                                decay = decay, pool_choice=pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, time_length, 
                                                time_window_size=moving_window[1], stride=stride[1], 
                                                decay = decay, pool_choice=pooling_choice)
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_dim * num_windows * num_node, 2*hidden_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2*hidden_dim, 2*hidden_dim)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(2*hidden_dim, hidden_dim)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc4', nn.Linear(hidden_dim, n_class)),
        ]))

    def forward(self, X):
        # print(X.size())
        bs, tlen, num_node, dimension = X.size() ### tlen = 1

        ### Graph Generation
        A_input = torch.reshape(X, [bs*tlen*num_node, dimension, 1])
        A_input_ = self.nonlin_map(A_input)
        A_input_ = torch.reshape(A_input_, [bs*tlen*num_node, -1])
        A_input_ = self.nonlin_map2(A_input_)
        A_input_ = torch.reshape(A_input_, [bs, tlen, num_node, -1])
        # print('A_input size is ', A_input_.size())

        ## positional encoding before mapping starting
        X_ = torch.reshape(A_input_, [bs, tlen, num_node, -1])
        X_ = torch.transpose(X_,1,2)
        X_ = torch.reshape(X_,[bs*num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = torch.reshape(X_,[bs,num_node, tlen, -1])
        X_ = torch.transpose(X_,1,2)
        A_input_ = X_

        ## positional encoding before mapping ending
        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)

        features1 = torch.reshape(MPNN_output1, [bs, -1])
        features2 = torch.reshape(MPNN_output2, [bs, -1])

        features = torch.cat([features1,features2],-1)
        features = self.fc(features)
        return features


#### Best for HAR
class FC_STGNN_HAR(nn.Module):
    def __init__(self, indim_fea, Conv_out, lstmhidden_dim, lstmout_dim, conv_kernel, hidden_dim, 
                time_length, num_node, num_windows, moving_window, stride, decay, pooling_choice, n_class):
        super(FC_STGNN_HAR, self).__init__()
        # graph_construction_type = args.graph_construction_type
        # 非线性映射模块，用于特征提取
        self.nonlin_map = Feature_extractor_1DCNN_HAR_SSC(1, lstmhidden_dim, lstmout_dim, kernel_size=conv_kernel)
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim*Conv_out, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim)
        )   # 180 → 32
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(2*hidden_dim, 0.1, max_len=5000)
        
        # 图构建和聚合：图卷积池化MPNN模块
        self.MPNN1 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, time_length, 
                                                time_window_size=moving_window[0], stride=stride[0], 
                                                decay = decay, pool_choice=pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, time_length, 
                                                time_window_size=moving_window[1], stride=stride[1], 
                                                decay = decay, pool_choice=pooling_choice)       
        # FC Graph Convolution
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_dim * num_windows * num_node, 2*hidden_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2*hidden_dim, 2*hidden_dim)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(2*hidden_dim, hidden_dim)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc4', nn.Linear(hidden_dim, n_class)),
        ]))

    def forward(self, X):
        bs, tlen, num_node, dimension = X.size()                        # 100, 2, 9, 64

        ### Graph Generation
        A_input = torch.reshape(X, [bs*tlen*num_node, dimension, 1])    # [1800, 64, 1]
        A_input_ = self.nonlin_map(A_input)                             # [1800, 18, 10]
        A_input_ = torch.reshape(A_input_, [bs*tlen*num_node,-1])       # [1800, 180]
        A_input_ = self.nonlin_map2(A_input_)                           # [1800, 32]
        A_input_ = torch.reshape(A_input_, [bs, tlen,num_node,-1])      # [100, 2, 9, 32]

        ## positional encoding before mapping starting
        X_ = torch.reshape(A_input_, [bs,tlen,num_node, -1])            # [100, 2, 9, 32]
        X_ = torch.transpose(X_, 1, 2)
        X_ = torch.reshape(X_, [bs*num_node, tlen, -1])                 # # [100, 2, 9, 32]
        X_ = self.positional_encoding(X_)
        X_ = torch.reshape(X_, [bs, num_node, tlen, -1])
        X_ = torch.transpose(X_, 1, 2)
        A_input_ = X_

        ## positional encoding before mapping ending
        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)

        features1 = torch.reshape(MPNN_output1, [bs, -1])
        features2 = torch.reshape(MPNN_output2, [bs, -1])
        features = torch.cat([features1,features2], -1)
        
        features = self.fc(features)
        return features


#### Best for SSC
class FC_STGNN_SSC(nn.Module):
    def __init__(self, indim_fea, Conv_out, lstmhidden_dim, lstmout_dim, conv_kernel,hidden_dim, 
                time_length, num_node, num_windows, moving_window,stride,decay, pooling_choice, n_class,dropout):
        super(FC_STGNN_SSC, self).__init__()
        self.nonlin_map = Feature_extractor_1DCNN_HAR_SSC(1, lstmhidden_dim, lstmout_dim,kernel_size=conv_kernel,dropout=dropout)
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim*Conv_out, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim)
        )

        self.positional_encoding = PositionalEncoding(2*hidden_dim,0.1,max_len=5000)

        self.MPNN1 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, time_length, 
                                                time_window_size=moving_window[0], stride=stride[0], 
                                                decay = decay, pool_choice=pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, time_length, 
                                                time_window_size=moving_window[1], stride=stride[1], 
                                                decay = decay, pool_choice=pooling_choice)
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_dim * num_windows * num_node, 2*hidden_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2*hidden_dim, 2*hidden_dim)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(2*hidden_dim, hidden_dim)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc4', nn.Linear(hidden_dim, n_class)),
        ]))

    def forward(self, X):
        # print(X.size())
        bs, tlen, num_node, dimension = X.size()

        ### Graph Generation
        A_input = torch.reshape(X, [bs*tlen*num_node, dimension, 1])
        A_input_ = self.nonlin_map(A_input)
        A_input_ = torch.reshape(A_input_, [bs*tlen*num_node,-1])
        A_input_ = self.nonlin_map2(A_input_)
        A_input_ = torch.reshape(A_input_, [bs, tlen,num_node,-1])

        ## positional encoding before mapping starting
        X_ = torch.reshape(A_input_, [bs,tlen,num_node, -1])
        X_ = torch.transpose(X_,1,2)
        X_ = torch.reshape(X_,[bs*num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = torch.reshape(X_,[bs,num_node, tlen, -1])
        X_ = torch.transpose(X_,1,2)
        A_input_ = X_

        ## positional encoding before mapping ending
        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)

        features1 = torch.reshape(MPNN_output1, [bs, -1])
        features2 = torch.reshape(MPNN_output2, [bs, -1])

        features = torch.cat([features1,features2],-1)
        features = self.fc(features)
        return features
