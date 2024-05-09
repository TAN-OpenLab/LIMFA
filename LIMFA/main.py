import torch

from model.LIMFA_NET import  LIMFA_Net


import argparse
import os
import sys
from string import digits

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def run(dataset, start_epoch):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=dataset,
                        choices=['weibo', '4charliehebdo','4ferguson','4germanwings-crash','4ottawashooting', '4sydneysiege'],
                        help='The name of dataset')
    parser.add_argument('--epochs', type=int, default= 200, help='The number of epochs to run')
    parser.add_argument('--start_epoch', type=int, default=start_epoch, help='Continue to train')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--num_posts', type=int, default=50)

    parser.add_argument('--data_syn_layer', type=int, default=2, help=' number layer for data synthsizing')
    parser.add_argument('--text_embedding', type =tuple, default=(768,200), help=' reduce the dimension of the text vector')
    parser.add_argument('--encoder_pars', type=tuple, default=(1,200, 2, 200, 0), help='num_layers, f_in, n_head,f_out,dropout')
    parser.add_argument('--GIN_pars', type=tuple, default=('mean', 'mean', 2, 2, 3, 20), help='aggregation_op, readout_op,'
                                                                             ' num_aggregation_layers, mlp_num_layers, num_features, hidden-dim')
    parser.add_argument('--Temp_MLP', type=tuple,default=(50,20), help='num_nodes, hidden-dim')
    parser.add_argument('--loss_func_list', type=tuple, default=(50, 20), help='num_nodes, hidden-dim')
    parser.add_argument('--lr', type=tuple, default=1e-3, help='lr')
    parser.add_argument('--dropout', type=float, default=0.0, help='attention dropout in GAT')
    parser.add_argument('--weight_decay', type=float, default=10, help='weight_decay')
    parser.add_argument('--patience', type=int, default=30, help='early stopping')
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--save_dir', type=str, default=r'E:\WDK_workshop\PCL_rumor_detection\result', help='Directory name to save the model')
    parser.add_argument('--model_name', type=str, default=r'MCSFD_centerL2_nodomain2', help='Directory name to save the GAN')
    parser.add_argument('--data_dir', default=os.path.join(r'E:\WDK_workshop\PCL_rumor_detection\data', dataset),type=str)
    # parser.add_argument('--data_dir', default=os.path.join(r'E:\WDK_workshop\SRGRD\data\ch-9\data_withdomain', dataset), type=str)

    if dataset == 'weibo':
        target_data = 'weibo2021'
    elif dataset == 'weibo2021':
        target_data = 'weibo'
    else:
        target_data = dataset.strip(digits)
    parser.add_argument('--target_domain', default=r'E:\WDK_workshop\PCL_rumor_detection\data\raw_data', type=str)
    # parser.add_argument('--target_domain', default=r'E:\WDK_workshop\SRGRD\data\ch-9\data_withdomain', type=str)
    parser.add_argument('--data_eval', default='', type=str)

    args = parser.parse_args()
    print(args)

    torch.set_default_dtype(torch.float32)
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

    model = LIMFA_Net(args, device)


    if not os.path.exists(os.path.join(args.save_dir, args.model_name)):
        os.mkdir(os.path.join(args.save_dir,  args.model_name))
    if not os.path.exists(os.path.join(args.save_dir, args.model_name, args.dataset)):
        os.mkdir(os.path.join(args.save_dir, args.model_name, args.dataset))

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



    model.train_epoch(args.data_dir, start_epoch)
    model.test(args.target_domain, target_data)
    print(" [*] Training finished!")



torch.manual_seed(6)
if __name__ == '__main__':

    datasets = [ '4charliehebdo', '4ferguson', '4germanwings-crash', '4ottawashooting', '4sydneysiege']
    # datasets = ['8教育考试', '8军事', '8科技', '8社会生活', '8文体娱乐', '8医药健康', '8灾难事故', '8政治']
    # datasets = ['4charliehebdo', '4ferguson', '4germanwings-crash', '4ottawashooting', '4sydneysiege', 'weibo','weibo2021']
    start_epoch = 72
    for dataset in datasets:
        print(dataset)
        run(dataset, start_epoch)

