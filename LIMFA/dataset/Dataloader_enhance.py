# -*-coding:utf-8-*-
"""
@Project    : LIMFA
@Time       : 2022/4/5 14:36
@Author     : Danke Wu
@File       : Dataloader_enhance.py
"""

import os
from dataset.Mydataset_enhance import MyDataset
from torch.utils.data import DataLoader
import math


def seperate_dataloader(datapath, dataset, batch, num_worker, num_nodes):

    nonrumor_path = os.path.join(datapath,dataset, 'nonrumor')
    nonrumor_files = os.listdir(nonrumor_path)
    nonrumor_dataset = MyDataset(nonrumor_path, nonrumor_files, num_nodes)
    rumor_path = os.path.join(datapath, dataset,'rumor')
    rumor_files = os.listdir(rumor_path)
    rumor_dataset = MyDataset(rumor_path, rumor_files, num_nodes)
    #DDT
    assert batch % 2 ==0 or batch % 4 ==0
    # rvsn = len(rumor_files) / len(nonrumor_files)
    # if rvsn >1 :
    #     n_rumor = int(batch / 2)
    # elif rvsn >= 0.5:
    #     n_rumor = round(rvsn * batch / 2)
    # elif rvsn < 0.5:
    #     n_rumor = int(batch / 4)
    # n_nonrumor = int(batch / 2)

    n_nonrumor, n_rumor = int(batch / 2), int(batch / 2)

    # #InF
    # rvsn = len(rumor_files) / (len(nonrumor_files) + len(rumor_files))
    # n_nonrumor,n_rumor = batch - math.ceil(batch * rvsn), math.ceil(batch * rvsn)

    nonrumor_loader = DataLoader(nonrumor_dataset,
                              batch_size= n_nonrumor,
                              shuffle=True,
                              num_workers=num_worker,
                              drop_last=False,
                              pin_memory=True)
    rumor_loader = DataLoader(rumor_dataset,
                              batch_size= n_rumor,
                              shuffle=True,
                              num_workers=num_worker,
                              drop_last=False,
                              pin_memory=True)

    return nonrumor_loader, rumor_loader


def imbalence_dataloader(datapath, dataset, batch, num_worker, num_nodes):

    nonrumor_path = os.path.join(datapath,dataset, 'nonrumor')
    nonrumor_files = os.listdir(nonrumor_path)
    rumor_path = os.path.join(datapath, dataset,'rumor')
    rumor_files = os.listdir(rumor_path)
    rumor_files.extend(nonrumor_files)

    imbdataset = MyDataset(rumor_path, rumor_files, num_nodes)

    data_loader = DataLoader(imbdataset,
                              batch_size= batch,
                              shuffle=True,
                              num_workers=num_worker,
                              drop_last=False,
                              pin_memory=True)

    return data_loader


def normal_dataloader(datapath, dataset, batch_size, num_worker, num_nodes):

    path = os.path.join(datapath, dataset)
    files = os.listdir(path)
    dataset = MyDataset(path, files, num_nodes)
    if dataset =='':
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_worker,
                                drop_last=False,
                                pin_memory=True)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_worker,
                                drop_last=False,
                                pin_memory=True)


    return dataloader