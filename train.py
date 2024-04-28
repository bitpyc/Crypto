import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import numpy as np
import time
from data_process import CryptoDataset, EarlyStopping, adjust_learning_rate
from sklearn import metrics
from torch.utils.data import DataLoader
from models import Dlinear


class TrainerCls():
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Dlinear': Dlinear
        }
        self.device = args.device
        self.model = self.model_dict[self.args.model](self.args).float().to(self.device)

    def _get_data(self, flag):
        data_set = CryptoDataset(self.args, flag)
        if flag == "test":
            data_loader = DataLoader(data_set, batch_size=1)
        else:
            data_loader = DataLoader(data_set, batch_size=self.args.batch_size)
        return data_set, data_loader

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoint_dir, setting)

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        tot_index = self.args.train_epochs * len(train_loader)
        index_now = 0
        index_ct = 0
        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, target) in enumerate(train_loader):
                index_ct += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                target = target.long().to(self.device)

                outputs = self.model(batch_x, None)
                outputs = outputs.reshape(-1, outputs.shape[-1]) # (batch_size * sequence_length, classes)
                target = target.reshape(-1) # (batch_size * sequence_length)
                loss = criterion(outputs, target)
                train_loss.append(loss.item())

                if (i + 1) % self.args.report_interval == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / (index_ct - index_now)
                    left_time = speed * (tot_index - index_ct)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    index_now = index_ct
                    time_now = time.time()

                loss.backward()
                optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy, val_f1, val_confusion = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy, test_f1, test_confusion = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Vali f1: {5:.3f} Test Loss: {6:.3f} Test Acc: {7:.3f} Test f1: {8:.3f}"
                .format(epoch + 1, len(train_loader), train_loss, vali_loss, val_accuracy, val_f1, test_loss, test_accuracy, test_f1))
            print("val confusion metrix:\n{}".format(val_confusion))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            '''
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(optimizer, epoch + 1, self.args)
            '''

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, target) in enumerate(vali_loader):
                # batch_x: input data
                # batch_y: label_length + predict horizon
                batch_x = batch_x.float().to(self.device)
                target = target.long().to(self.device)

                outputs = self.model(batch_x, None)

                outputs = outputs.reshape(-1, outputs.shape[-1]).detach().cpu()
                target = target.reshape(-1).detach().cpu()

                loss = criterion(outputs, target)
                total_loss.append(loss.item())

                preds.append(outputs)
                trues.append(target)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.softmax(preds, dim=-1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()

        accuracy = metrics.accuracy_score(trues, predictions)
        f1 = metrics.f1_score(trues, predictions, average="macro")
        confusion = metrics.confusion_matrix(trues, predictions)

        self.model.train()
        return total_loss, accuracy, f1, confusion

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(self.args.checkpoint_file))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, target) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                target = target.long().to(self.device)

                outputs = self.model(batch_x, None)

                outputs = outputs.reshape(-1, outputs.shape[-1]).detach().cpu()
                target = target.reshape(-1).detach().cpu()

                preds.append(outputs)
                trues.append(target)

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        probs = torch.softmax(preds, dim=-1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = metrics.accuracy_score(trues, predictions)
        f1 = metrics.f1_score(trues, predictions, average="macro")
        confusion = metrics.confusion_matrix(trues, predictions)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        result_info = 'accuracy: {}, f1: {}\nconfusion metrix:\n{}'.format(accuracy, f1, confusion)
        print(result_info)
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write(result_info)
        f.write('\n')
        f.write('\n')
        f.close()

        f = open("training result.txt", 'a')
        f.write(setting + "  \n")
        f.write(result_info)
        f.write('\n')
        f.write('\n')
        f.close()
        return


class InferenceCls():
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Dlinear': Dlinear
        }
        self.device = args.device
        self.model = self.model_dict[self.args.model](self.args).float().to(self.device)
        self.model.load_state_dict(torch.load(args.checkpoint_file))

    # data: (sequence_length, feature_size)
    def inference(self, data):
        data = torch.Tensor(data).unsqueeze(0).float().to(self.device)
        outputs = self.model(data, None)
        outputs = outputs.squeeze(0).detach().cpu()
        probs = torch.softmax(outputs, dim=-1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        return preds

    def _get_data(self):
        data_set = CryptoDataset(self.args, "test")
        data_loader = DataLoader(data_set, batch_size=1)
        return data_set, data_loader

    def test(self):
        test_data, test_loader = self._get_data()
        time_st = time.time()
        for i, (batch_x, target) in enumerate(test_loader):
            data = batch_x[0].numpy()
            pred = self.inference(data)
        time_ed = time.time()
        print("total time spend:{}s, time per index:{}s/index".format(time_ed - time_st, (time_ed - time_st) / len(test_loader)))