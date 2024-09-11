import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from data_loader_MSGNet import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Flight

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, args):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"] 
        print(X_train.shape, y_train.shape)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train.float()   # x_data:[5881, 9, 128]
            self.y_data = y_train.long()    # y_data:[5881]

        self.len = X_train.shape[0]         # len(X_train): 5881
        shape = self.x_data.size()          
        self.x_data = self.x_data.reshape(shape[0],shape[1],args.time_denpen_len, args.patch_size)  # [5881, 9, 2, 64]
        self.x_data = torch.transpose(self.x_data, 1, 2)
    # 返回指定索引处的样本及其对应的标签
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    # 返回数据集中样本的数量
    def __len__(self):
        return self.len


# 数据生成器，用于加载训练集、验证集和测试集，并将它们转换为PyTorch的DataLoader对象
def data_generator(data_path, args):

    train_dataset = torch.load(os.path.join(data_path, "train.pt"), weights_only=True)
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"), weights_only=True)
    test_dataset = torch.load(os.path.join(data_path, "test.pt"), weights_only=True)

    train_dataset = Load_Dataset(train_dataset, args)   # x_train:[5881, 2, 9, 64], y_train:[5881]
    valid_dataset = Load_Dataset(valid_dataset, args)   # x_valid:[1173, 2, 9, 64], y_valid:[1173]
    test_dataset = Load_Dataset(test_dataset, args)     # x_test:[2947, 2, 9, 128], y_test:[2947]

    x = []
    for i, data in enumerate(train_dataset):
        x_train, y_train = data
        x.append(x_train)
    print("i: ", i)
    print("len(x): ", len(x))
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=args.batch_size,
                                        shuffle=True, 
                                        drop_last=args.drop_last, 
                                        num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, 
                                        batch_size=args.batch_size,
                                        shuffle=False, 
                                        drop_last=args.drop_last, 
                                        num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=args.batch_size,
                                        shuffle=False, 
                                        drop_last=False, 
                                        num_workers=0)
    return train_loader, valid_loader, test_loader


def data_provider_ETT(args, flag):
    Data = data_dict[args.data]
    # time features encoding, options: [timeF, fixed, learned]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred

    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(root_path=args.root_path,
                    data_path=args.data_path,
                    flag=flag,
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features,
                    target=args.target,
                    timeenc=timeenc,
                    freq=freq,
                    seasonal_patterns = args.seasonal_patterns
    )
    print(flag, len(data_set))
    # data_x:(34560, 7), data_y:(34560, 7), data_stamp:(34560, 4)

    # batch_x_data = []
    # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_set):
    #     batch_x_data.append(batch_x)
    #     # print(batch_x_data, batch_y_data, batch_x_mark_data, batch_y_mark_data)
    #     print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
    # print(i)
    # print(batch_x_data.shape)
    
    data_loader = DataLoader(data_set,
                            batch_size=batch_size,
                            shuffle=shuffle_flag,
                            num_workers=args.num_workers,
                            drop_last=drop_last
    )
    
    # batch_x_data = []
    # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_set):
    #     batch_x_data.append(batch_x)
    #     print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
    # print(i)
    # print(batch_x_data.shape)
    
    
    return data_set, data_loader