# -*-coding:utf-8-*-

import os, re
from MultiDomain_Baselines.Dataset import MyDataset
from torch.utils.data import DataLoader
import math


def seperate_dataloader(datapath, traintype, batch, num_worker, num_nodes,domain_dict):

    nonrumor_path = os.path.join(datapath, traintype, 'nonrumor')
    nonrumor_files = os.listdir(nonrumor_path)
    nonrumor_dataset = MyDataset(nonrumor_path, nonrumor_files, num_nodes, domain_dict)
    rumor_path = os.path.join(datapath, traintype,'rumor')
    rumor_files = os.listdir(rumor_path)
    rumor_dataset = MyDataset(rumor_path, rumor_files, num_nodes,domain_dict)
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
                              drop_last=True,
                              pin_memory=True)
    rumor_loader = DataLoader(rumor_dataset,
                              batch_size= n_rumor,
                              shuffle=True,
                              num_workers=num_worker,
                              drop_last=False,
                              pin_memory=True)

    return nonrumor_loader, rumor_loader


def normal_dataloader(datapath, traintype, batch_size, num_worker, num_nodes, domain_dict):


    path = os.path.join(datapath,  traintype)
    files = os.listdir(path)
    dataset = MyDataset(path, files, num_nodes, domain_dict)

    dataloader = DataLoader(dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_worker,
                               drop_last=False,
                               pin_memory=True)
    return dataloader