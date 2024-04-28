import os
import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


class CryptoDataset(Dataset):
    def __init__(self, configs, flag="train", scale=True):
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = configs.target
        self.scale = scale
        self.task = configs.task

        self.root_path = configs.root_path
        self.data_path = configs.data_path
        self.threshold = configs.class_threshold
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_data = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_data = df_data.drop(labels=["start_time"], axis=1)

        start_of_dataset = [0, 18 * 3600 - self.seq_len, 21 * 3600 - self.seq_len]
        end_of_dataset = [18 * 3600, 21 * 3600, 24 * 3600]
        border1 = start_of_dataset[self.set_type]
        border2 = end_of_dataset[self.set_type]

        if self.scale:
            train_data = df_data[start_of_dataset[0]:end_of_dataset[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        target_data = self.get_target_data(df_data)
        self.data_x = data[border1:border2]
        self.data_y = target_data[border1:border2]

    def get_target_data(self, df_data):
        diff_percentage = np.zeros_like(df_data[self.target])
        diff_percentage[1:] = (df_data[self.target].values[1:] - df_data[self.target].values[:-1]) / (df_data[self.target].values[:-1] + 1e-10)
        '''
        sr = pd.Series(np.abs(diff_percentage))
        print(sr.describe())
        sr.plot.kde()
        plt.show()
        plt.close()
        '''
        # 0: fall, 1:unchanged, 2:rise
        target_data = np.array([1 if abs(x) < self.threshold else 2 * int(x > 0) for x in diff_percentage], dtype=np.int64)
        return target_data

    def __getitem__(self, item):
        s_begin = item
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss