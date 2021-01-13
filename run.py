# -*- coding: utf-8 -*-
import os
from datetime import datetime

import torch
import pickle
import argparse
import numpy as np
import torch.utils.data

from model import MG_MC
from multiGraph import multiGraph
from predication import train, test


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='MG_MC')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--embed_dim', type=int, default=32, help='embedding dimension')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--N', type=int, default=30000, help='L0 parameter')
    parser.add_argument('--droprate', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=256, help='input batch size for testing')
    parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    parser.add_argument("--use_cuda", type=bool, default=True, help='use cuda or not')
    args = parser.parse_args()

    print('-------------------- Hyperparams --------------------')
    print('time: ' + str(datetime.now()))
    print('Dataset: ' + args.dataset)
    print('N: ' + str(args.N))
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))
    print('use_cuda: ' + str(args.use_cuda))

    device = torch.device("cuda" if args.use_cuda else "cpu")

    if args.use_cuda:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = True
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    embed_dim = args.embed_dim
    data_path = './datasets/' + args.dataset

    with open(data_path + '/_allData.p', 'rb') as meta:
        u2e, i2e, u_train, i_train, r_train, u_test, i_test, r_test, u_adj, i_adj = pickle.load(meta)

    with open(data_path + '/' + args.dataset + '.p', 'rb') as meta:
        u_neighs, i_neighs = pickle.load(meta)

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(u_train), torch.LongTensor(i_train),
                                              torch.FloatTensor(r_train))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(u_test), torch.LongTensor(i_test),
                                             torch.FloatTensor(r_test))

    _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, num_workers=16)

    u_embed, i_embed = multiGraph(u2e, i2e, u_neighs, i_neighs, u_adj, i_adj, embed_dim,
                                  droprate=args.droprate, device=device, weight_decay=args.weight_decay)

    #  model
    model = MG_MC(u_embed, i_embed, embed_dim, args.N, droprate=args.droprate, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rmse_mn = np.inf
    mae_mn = np.inf
    endure_count = 0
    for epoch in range(1, args.epochs + 1):
        train(model, _train, optimizer, epoch, rmse_mn, mae_mn, device)
        rmse, mae = test(model, _test, device)

        if mae_mn > mae:
            rmse_mn = rmse
            mae_mn = mae
            endure_count = 0
        else:
            endure_count += 1
        print("<Test> RMSE：%.5f,MAE：%.5f" % (rmse, mae))

        if endure_count > 30:
            break

    print('The best RMSE/MAE：%.5f/%.5f' % (rmse_mn, mae_mn))


if __name__ == "__main__":
    main()
