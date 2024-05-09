# -*-coding:utf-8-*-
"""
@Project    : LIMFA
@Time       : 2022/5/12 10:43
@Author     : Danke Wu
@File       : EANN_NET.py
"""
# -*-coding:utf-8-*-

import random

import threadpoolctl
import torch
import torch.autograd as autograd
import torch.nn as nn
from MultiDomain_Baselines.Dataloader import seperate_dataloader, normal_dataloader
from Loss_Functions.evaluation import *
from Loss_Functions.loss_wrapper import LossWrapper_center
from MultiDomain_Baselines.eann import CNN_Fusion
from MultiDomain_Baselines.bdann import Transformer_Fusion
from model.Model_layers import LambdaLR
from Loss_Functions.loss_wrapper import Center_Loss
import os, sys
import time
from itertools import cycle as CYCLE


class EANN_NET(object):
    def __init__(self, args, device):
        # parameters

        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.batch_size = args.batch_size
        self.num_posts = args.num_posts
        self.domain_num, self.filter_num, self.f_in, self.emb_dim, self.dropout = args.eann_pars

        self.num_worker = args.num_worker
        self.weight_decay = args.weight_decay
        self.device = device
        self.checkpoint = args.start_epoch
        self.M_lr = args.M_lr
        self.patience = args.patience
        self.dataset = args.dataset
        self.model_path = os.path.join(args.save_dir, args.model_name, args.dataset)
        self.center = args.CENTER

        #=====================================load rumor_detection model================================================

        # self.model= CNN_Fusion(self.domain_num, self.filter_num,self.f_in, self.emb_dim, self.dropout) #EANN
        self.model = Transformer_Fusion(self.domain_num, self.filter_num,self.f_in, self.emb_dim, self.dropout) #BDANN
        #

        self.model.to(self.device)
        print(self.model)

        # =====================================load loss function================================================
        self.celoss = nn.CrossEntropyLoss()
        if self.center:
            self.centerloss = Center_Loss()

        #self.optimizer = torch.optim.Adam([{'stu_text_content':self.stu_text_content.parameters()},
                                          # {'stu_text_style': self.stu_text_style.parameters()},
                                          # {'stu_structure': self.stu_structure.parameters()},
                                          # {'stu_temporal': self.stu_temporal.parameters()},
                                          # {'content_reconduction': self.reconduction.parameters()}
                                          # ], lr= self.lr, betas=(self.b1, self.b2))
        self.optimizer = torch.optim.SGD([{'params': self.model.parameters(),'lr': self.M_lr, 'momentum' : 0.9, 'weight_decay':1e-2}])
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                              lr_lambda=LambdaLR(self.epochs, self.start_epoch,
                                                                               decay_start_epoch= self.weight_decay).step)

        torch.autograd.set_detect_anomaly(True)

    def train(self, datapath, start_epoch, domain_dict):

        nonrumor_loader, rumor_loader = seperate_dataloader(datapath, 'train', self.batch_size, self.num_worker,
                                                            self.num_posts, domain_dict)
        val_loader = normal_dataloader(datapath, 'val', self.batch_size, self.num_worker, self.num_posts,domain_dict)

        # ==================================== train and val dataGAN with model=========================================
        acc_all_check = 0
        loss_class_check, loss_domain_check = 100, 100
        start_time = time.clock()
        patience = self.patience
        for epoch in range(start_epoch, self.epochs):

            train_loss, acc = self.train_batch(epoch, nonrumor_loader, rumor_loader)
            self.lr_scheduler.step()
            print(train_loss, acc)
            with torch.no_grad():
                val_c_loss,val_d_loss, val_acc = self.evaluation(val_loader)
            end_time = time.clock()
            print(val_c_loss, val_d_loss, val_acc)

            if val_c_loss <= loss_class_check:# or
                acc_all_check = val_acc['Acc_all']
                loss_class_check = val_c_loss
                loss_domain_check = val_d_loss
                patience = self.patience
                self.checkpoint = epoch
                self.save(self.model_path, epoch, source=True)

            patience -= 1

            if not patience:
                break

        # ==================================== test model with model================================================
        with torch.no_grad():
            test_loader = normal_dataloader(datapath,'test',self.batch_size, self.num_worker, self.num_posts,domain_dict)

            start_epoch = self.load(self.model_path, self.checkpoint, source= True)

            test_c_loss, test_d_loss, test_acc = self.evaluation(test_loader)

            with open(os.path.join(self.model_path, 'predict.txt'), 'a') as f:
               f.write( '\t'.join(list(test_acc.keys())) + '\n' + '\t'.join(map(str,list(test_acc.values()))) + '\n')

    def train_batch(self, epoch, nonrumor_loader, rumor_loader):

        train_loss_value = 0
        acc_value =0

        # if len(rumor_loader) > len(nonrumor_loader):
        #     iterloader = zip(CYCLE(nonrumor_loader), rumor_loader)
        # # elif len(rumor_loader) / len(nonrumor_loader) >= 0.5:
        # #     iterloader = zip(nonrumor_loader, rumor_loader)
        # else:
        #     iterloader = zip(nonrumor_loader, CYCLE(rumor_loader))

        iterloader = zip(nonrumor_loader, rumor_loader)

        for iter, (Nonrumors, Rumors) in enumerate(iterloader):
            xn, yn, Dn = Nonrumors
            xr, yr, Dr = Rumors
            xn = xn.to(self.device)
            yn = yn.to(self.device)
            xr = xr.to(self.device)
            yr = yr.to(self.device)
            Dr = Dr.to(self.device)
            Dn = Dn.to(self.device)

            x = torch.cat((xn, xr) ,dim=0)
            y = torch.cat((yn, yr), dim=0)
            D = torch.cat((Dn, Dr), dim=0)
            # ====================================train Model============================================
            self.model.train()

            self.optimizer.zero_grad()

            # D_one_hot = torch.zeros(self.batch_size, self.domain_num, device=D.device,dtype=torch.long).scatter_(1, D, 1)

            c_preds, d_preds, text = self.model(x)
            c_loss = self.celoss(c_preds, y)
            d_loss = self.celoss(d_preds, D)
            if self.center == True:
                center_loss = self.centerloss(text, y, c_preds)
                loss = c_loss + d_loss + center_loss
            else:
                center_loss = torch.zeros(1)
                loss = c_loss + d_loss

            loss.backward()
            self.optimizer.step()

            train_loss_value += loss.item()

            pred = torch.max(c_preds,dim=1)[1]
            acc = (pred == y).sum() / len(y)
            acc_value += acc.item()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [c loss: %f][d loss: %f] [ceterloss: %f ][model acc: %f]"
                % (
                    epoch,
                    self.epochs,
                    iter,
                    len(nonrumor_loader),
                    c_loss.item(),
                    d_loss.item(),
                    center_loss.item(),
                    acc.item()
                )
            )

        train_loss_value = round(train_loss_value /(iter+1),4)
        acc_value = round(acc_value /(iter+1),4)
        return train_loss_value, acc_value

    def evaluation(self, dataloader):

        mean_c_loss, mean_d_loss,acc_value = 0, 0, 0
        Conter_Item = ['TP1', 'FP1', 'FN1', 'TN1', 'TP2', 'FP2', 'FN2', 'TN2']
        Conter = {}
        for item in Conter_Item:
            Conter[item] = 0
        num_sample = 0
        self.model.eval()

        for iter, sample in enumerate(dataloader):
            x, y, D = sample
            x = x.to(self.device)
            y = y.to(self.device)
            D = D.to(self.device)

            c_preds, d_preds, text = self.model(x)
            c_loss = self.celoss(c_preds, y)
            d_loss = self.celoss(d_preds, D)
            loss = c_loss + d_loss
            mean_c_loss += c_loss.item()
            mean_d_loss += d_loss.item()
            preds = c_preds.data.max(1)[1].cpu()

            (tp1, fn1, fp1, tn1, tp2, fn2, fp2, tn2) = count_2class(preds, y.cpu())
            Conter['TP1'] += tp1
            Conter['FN1'] += fn1
            Conter['FP1'] += fp1
            Conter['TN1'] += tn1
            Conter['TP2'] += tp2
            Conter['FN2'] += fn2
            Conter['FP2'] += fp2
            Conter['TN2'] += tn2

            num_sample += len(y)

        Acc_dict = evaluationclass(Conter, num_sample)
        mean_c_loss = round(mean_c_loss / (iter + 1), 4)
        mean_d_loss = round(mean_d_loss / (iter + 1), 4)

        return mean_c_loss, mean_d_loss, Acc_dict

    def test(self, datapath, dataset, domain_dict):

        test_loader = normal_dataloader(datapath, dataset, self.batch_size, self.num_worker, self.num_posts, domain_dict)
        with torch.no_grad():
            start_epoch = self.load(self.model_path, self.checkpoint, source=False)
            acc_test_dict = self.transfer_test(test_loader)
            print(acc_test_dict.items())
            with open(os.path.join(self.model_path, 'predict.txt'), 'a') as f:
                f.write('target doamin' + '\t' + str(start_epoch) + '\n' +
                        '\t'.join(list(acc_test_dict.keys())) + '\n' + '\t'.join(
                    map(str, list(acc_test_dict.values()))) + '\n')

        return 0

    def transfer_test(self, dataloader):

        Conter_Item = ['TP1', 'FP1', 'FN1', 'TN1', 'TP2', 'FP2', 'FN2', 'TN2']
        Conter = {}
        for item in Conter_Item:
            Conter[item] = 0
        num_sample = 0
        self.model.eval()

        for iter, sample in enumerate(dataloader):
            x, y, D = sample
            x = x.to(self.device)
            y_ = y.to(self.device)
            D = D.to(self.device)

            c_preds,_,_ = self.model(x)

            preds = c_preds.data.max(1)[1].cpu()

            (tp1, fn1, fp1, tn1, tp2, fn2, fp2, tn2) = count_2class(preds, y_)
            Conter['TP1'] += tp1
            Conter['FN1'] += fn1
            Conter['FP1'] += fp1
            Conter['TN1'] += tn1
            Conter['TP2'] += tp2
            Conter['FN2'] += fn2
            Conter['FP2'] += fp2
            Conter['TN2'] += tn2

            num_sample += len(y)

        Acc_dict = evaluationclass(Conter, num_sample)

        return Acc_dict


    def save(self, model_path, epoch, source = True):
        save_states = {
                       'model': self.model.state_dict(),
                       'optimizer': self.optimizer.state_dict(),
                       'checkpoint': epoch}
        torch.save(save_states, os.path.join(model_path, str(epoch) + '_model_states.pkl'))
        print('save classifer : %d epoch' % epoch)



    def load(self, model_path, checkpoint, source=True):
        states_dicts = torch.load( os.path.join(model_path, str(checkpoint) + '_model_states.pkl'))

        self.model.load_state_dict(states_dicts['model'])
        self.optimizer.load_state_dict(states_dicts['optimizer'])
        start_epoch = states_dicts['checkpoint']
        print("load epoch {} success!".format(start_epoch))

        return start_epoch


