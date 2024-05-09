# -*-coding:utf-8-*-
"""
@Project    : LIMFA
@Time       : 2022/5/11 10:06
@Author     : Danke Wu
@File       : Dataset.py
"""
import torch.utils.data as Data
import torch
import pickle
import os
from prefetch_generator import BackgroundGenerator
from torch_geometric.data import Data as geoData

class DataLoaderX(Data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

#filepath = 'D:\\学术相关\\007.CasCN-master\\dataset_weibo'
class MyDataset(Data.Dataset):
    def __init__(self, filepath, file_list, num_nodes, domain_dict):
        # 获得训练数据的总行
        self.filepath = filepath
        self.file_list = file_list
        self.number = len(self.file_list)
        self.num_nodes = num_nodes
        self.domain_dict= domain_dict

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        file = self.file_list[idx]
        #X: content matirx (N,768) ,XS:node degree matrix(N,3), A: adjection matrix (N,N) ,T: public _time_invterval(N,1) , y:label 0or1
        X, A, y, d = pickle.load(open(os.path.join(self.filepath, file), 'rb'), encoding='utf-8')

        D = torch.tensor(self.domain_dict[d], dtype=torch.long)

        self.number = len(X)
        X = torch.tensor(X, dtype=torch.float32)

        if torch.sum(torch.isinf(X)) >0 or torch.sum(torch.isnan(X))>0:
            print(file)

        #early detection
        if (X.size()[0] > self.num_nodes):
            X = X[:self.num_nodes, :]
            A = A[:self.num_nodes, :self.num_nodes]
        y = torch.tensor(y, dtype=torch.long)
        N, F = X.size()
        if (X==0).sum() == F*N :
            print(os.path.join(self.filepath, file))
        if torch.sum(X.sum(-1))==X.size()[0]:
            print(os.path.join(self.filepath, file))

        return X, y , D#,self.L[idx]

