import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
import Model
from args import args
from data_loader_HAR import data_generator


class Train():
    def __init__(self, args):
        self.train, self.valid, self.test = data_generator('../data/HAR/', args=args)
        self.args = args
        self.net = Model.FC_STGNN_HAR(args.patch_size,args.conv_out, args.lstmhidden_dim, args.lstmout_dim,
                                        args.conv_kernel,args.hidden_dim,args.time_denpen_len, args.num_sensor, 
                                        args.num_windows,args.moving_window,args.stride, args.decay, 
                                        args.pool_choice, args.n_class)
        self.net = self.net.cuda() if torch.cuda.is_available() else self.net
        self.loss_function = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.net.parameters())

    def cuda_(self, x):
        x = torch.Tensor(np.array(x))
        if torch.cuda.is_available():
            return x.cuda()
        else:
            return x

# 计算整个训练集的累计损失值
    def Train_batch(self):
        self.net.train()
        loss_ = 0
        for data, label in self.train:
            data = data.cuda() if torch.cuda.is_available() else data
            label = label.cuda() if torch.cuda.is_available() else label
            self.optim.zero_grad()
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss.backward()
            self.optim.step()
            loss_ = loss_ + loss.item()
        return loss_

# 训练模型：通过循环进行多个epoch的训练，每个epoch中训练一批数据，并在每个show_interval间隔时进行交叉验证。
    def Train_model(self):
        epoch = self.args.epoch
        cross_accu = 0
        test_accu_ = []
        prediction_ = []
        real_ = []
        for i in range(epoch):
            time0 = time.time()
            loss = self.Train_batch()   # 返回整个训练集的累计损失值
            if i % self.args.show_interval == 0:
                accu_val = self.Cross_validation()
                if accu_val > cross_accu:
                    cross_accu = accu_val
                    test_accu, prediction, real = self.Prediction()
                    print('In the {}th epoch, TESTING accuracy is {}%'.format(i, np.round(test_accu, 3)))
                    test_accu_.append(test_accu)
                    prediction_.append(prediction)
                    real_.append(real)
            print(i)
        print(epoch)
        combined_array = np.column_stack((array1, array2, array3))
        np.save('./experiment/test.npy',[test_accu_, prediction_, real_])
        # np.save('./experiment/{}.npy'.format(self.args.save_name),[test_accu_, prediction_, real_])


# 交叉验证
    def Cross_validation(self):
        self.net.eval()
        prediction_ = []
        real_ = []
        for data, label in self.valid:
            data = data.cuda() if torch.cuda.is_available() else data
            real_.append(label)
            prediction = self.net(data)
            prediction_.append(prediction.detach().cpu())
        prediction_ = torch.cat(prediction_,0)
        real_ = torch.cat(real_,0)
        prediction_ = torch.argmax(prediction_,-1)
        accu = self.accu_(prediction_, real_)
        return accu

# 计算预测值predicted与真实值real之间的准确率百分比
    def accu_(self, predicted, real):
        num = predicted.size(0)
        real_num = 0
        for i in range(num):
            if predicted[i] == real[i]:
                real_num+=1
        return 100*real_num/num

# 对测试数据集进行预测，计算预测准确性
    def Prediction(self):
        self.net.eval() # 设置模型为评估模式
        prediction_ = []
        real_ = []
        for data, label in self.test:
            data = data.cuda() if torch.cuda.is_available() else data
            real_.append(label)
            prediction = self.net(data) # 使用模型self.net对数据进行预测
            prediction_.append(prediction.detach().cpu())
        prediction_ = torch.cat(prediction_, 0)
        real_ = torch.cat(real_, 0)
        prediction_ = torch.argmax(prediction_, -1)
        
        accu = self.accu_(prediction_, real_)
        return accu, prediction_, real_


if __name__ == '__main__': 
    args = args()
    def args_config_HAR(args):
        args.epoch = 41
        args.k = 1
        args.window_sample = 128

        args.batch_size = 100
        args.decay = 0.7
        args.pool_choice = 'mean'
        args.moving_window = [2, 2]
        args.stride = [1, 2]
        args.lr = 1e-3
        
        args.conv_kernel = 6
        args.patch_size = 64
        args.time_denpen_len = int(args.window_sample / args.patch_size)
        args.conv_out = 10
        args.num_windows = 2

        args.conv_time_CNN = 6

        args.lstmout_dim = 18
        args.hidden_dim = 16
        args.lstmhidden_dim = 48

        args.num_sensor = 9
        args.n_class = 6
        return args


    args = args_config_HAR(args)
    train = Train(args)
    train.Train_model()
