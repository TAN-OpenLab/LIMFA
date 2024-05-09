# -*-coding:utf-8-*-
"""
@Project    : LIMFA
@Time       : 2022/5/11 11:10
@Author     : Danke Wu
@File       : main.py
"""
import torch
from MultiDomain_Baselines.MDFEND_NET import MDFEND_NET
from MultiDomain_Baselines.EANN_NET import EANN_NET


import argparse
import os, re
import sys

from string import digits

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def run(dataset, start_epoch):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=dataset,
                        choices=['weibo', '4charliehebdo','4ferguson','4germanwings-crash','4ottawashooting', '4sydneysiege'],
                        help='The name of dataset')
    parser.add_argument('--epochs', type=int, default= 200, help='The number of epochs to run')
    parser.add_argument('--start_epoch', type=int, default= start_epoch, help='Continue to train')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--num_posts', type=int, default=50)

    parser.add_argument('--mdfend_paras', type=tuple, default=(5, 3, 768, [200], 0.2), help=' domain_num, num_expert,emb_dim, mlp_dims, dropout')
    parser.add_argument('--eann_pars', type=tuple, default=(4, 5, 768, 200, 0.2),
                        help=' domain_num, filter_num, f_in, emb_dim,  dropout')
    parser.add_argument('--M_lr', type=tuple, default=5e-3, help='lr')
    parser.add_argument('--weight_decay', type=float, default=10, help='weight_decay')
    parser.add_argument('--patience', type=int, default=30, help='early stopping')

    parser.add_argument('--save_dir', type=str, default=r'E:\WDK_workshop\PCL_rumor_detection\result', help='Directory name to save the model')
    parser.add_argument('--model_name', type=str, default=r'EANN_center_64', help='Directory name to save the GAN')
    parser.add_argument('--data_dir', default=os.path.join(r'E:\WDK_workshop\PCL_rumor_detection\data\data_withdomain', dataset),
                        type=str)
    if dataset == 'weibo':
        target_data = 'weibo2021'
    elif dataset == 'weibo2021':
        target_data = 'weibo'
    else:
        target_data = dataset.strip(digits)
    parser.add_argument('--target_domain', default=r'E:\WDK_workshop\PCL_rumor_detection\data\raw_data_withdomain', type=str)
    parser.add_argument('--CENTER', default=True, type=bool)


    args = parser.parse_args()
    print(args)

    torch.set_default_dtype(torch.float32)
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

    # model = MDFEND_NET(args, device)
    model = EANN_NET(args, device)


    if not os.path.exists(os.path.join(args.save_dir, args.model_name)):
        os.mkdir(os.path.join(args.save_dir,  args.model_name))
    if not os.path.exists(os.path.join(args.save_dir, args.model_name, args.dataset)):
        os.mkdir(os.path.join(args.save_dir, args.model_name, args.dataset))

    data_start_epochs =0

    if os.path.exists(os.path.join(args.save_dir, args.model_name, args.dataset, str(args.start_epoch) + '_model_states.pkl')):
        model_path = os.path.join(args.save_dir, args.model_name, args.dataset)
        start_epoch = model.load(model_path, args.start_epoch)
        start_epoch = start_epoch + 1
    else:
        start_epoch = 0
        print("start from epoch {}".format(start_epoch))
        argsDict = args.__dict__
        with open(os.path.join(args.save_dir, args.model_name, args.dataset, 'setting.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')

    all_domians = set(['4charliehebdo', '4ferguson', '4germanwings-crash', '4ottawashooting', '4sydneysiege'])
    domians = all_domians - set([args.dataset])
    idx = 0
    #source domian train
    domian2idx_dict = {}
    for domian in domians:
        d = re.sub(r'[^A-Za-z\-]', '', domian)
        domian2idx_dict[d] = idx
        idx += 1
    model.train(args.data_dir, start_epoch, domian2idx_dict)


    #target domain test
    domian2idx_dict = {}
    d = re.sub(r'[^A-Za-z\-]', '', args.dataset)
    domian2idx_dict[d] = 4
    model.test(args.target_domain,target_data, domian2idx_dict)
    print(" [*] Training finished!")


torch.manual_seed(6)
if __name__ == '__main__':

    datasets = [ '4charliehebdo','4ferguson', '4germanwings-crash', '4ottawashooting', '4sydneysiege',]
    # datasets = ['4charliehebdo', '4ferguson', '4germanwings-crash', '4ottawashooting', '4sydneysiege', 'weibo','weibo2021']
    start_epoch = 3
    for dataset in datasets:
        print(dataset)
        run(dataset, start_epoch)





