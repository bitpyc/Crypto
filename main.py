import argparse
import os
from train import TrainerCls, InferenceCls
from data_process import CryptoDataset
import pandas as pd
import torch.cuda
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="datasets", help="data root")
    parser.add_argument("--data_path", type=str, default="top2_2023-12-16_to_2023-12-17_local.csv", help="data file name")

    parser.add_argument("--seq_len", type=int, default=720, help="length of look back window")
    parser.add_argument("--label_len", type=int, default=0, help="length of label sequence")
    parser.add_argument("--pred_len", type=int, default=30, help="length of predict horizon")

    parser.add_argument("--task", type=str, default="SeqCls", help="task type")
    parser.add_argument("--target", type=str, default="wap0", help="target variable")
    parser.add_argument("--classes", type=int, default=3, help="classification classes")
    parser.add_argument("--class_threshold", type=float, default=5e-5, help="threshold rate of unchanged class")

    parser.add_argument("--model", type=str, default="Dlinear")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="checkpoints dir")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoints/SeqCls_Dlinear_720_30/checkpoint.pth", help="checkpoints model for test or inference")
    parser.add_argument("--is_training", type=bool, default=True)
    parser.add_argument("--inference_only", type=bool, default=True)

    parser.add_argument("--batch_size", type=int, default=720)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--patience", type=int, default=9, help="early stop patience")
    parser.add_argument("--report_interval", type=int, default=50, help="interval of iteration speed report")

    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    args.use_gpu = True if args.use_gpu and torch.cuda.is_available() else False

    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Use GPU: cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')
        print('Use CPU')

    setting = '{}_{}_{}_{}'.format(args.task, args.model, args.seq_len, args.pred_len)
    print("settings: {}".format(setting))
    df_data = pd.read_csv(os.path.join(args.root_path, args.data_path))

    args.variable_size = df_data.shape[1] - 1
    trainer = TrainerCls(args)
    dataset, data_loader = trainer._get_data("test")
    labels = pd.Series(dataset.data_y)
    print(labels.value_counts())


    if args.inference_only:
        infer = InferenceCls(args)
        infer.test()
        '''
        inference_dataset = CryptoDataset(args, "test")
        data, label = inference_dataset.__getitem__(0)
        preds = infer.inference(data)
        print("inference accuracy={}".format(np.mean(preds == label)))
        '''
    elif args.is_training:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        trainer.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        trainer.test(setting)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        trainer.test(setting, test=1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
