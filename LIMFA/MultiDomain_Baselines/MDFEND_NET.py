# -*-coding:utf-8-*-
"""
@Project    : LIMFA
@Time       : 2022/5/11 10:13
@Author     : Danke Wu
@File       : MDFEND_NET.py
"""

import random

import threadpoolctl
import torch
import torch.autograd as autograd
import torch.nn as nn
from MultiDomain_Baselines.Dataloader import seperate_dataloader, normal_dataloader
from Loss_Functions.evaluation import *
from Loss_Functions.loss_wrapper import LossWrapper_center
from MultiDomain_Baselines.mdfend import MultiDomainFENDModel
from model.Model_layers import LambdaLR
import os, sys
import time
from itertools import cycle as CYCLE


class MDFEND_NET(object):
    def __init__(self, args, device):
        # parameters

        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.batch_size = args.batch_size
        self.num_posts = args.num_posts
        self.domain_num, self.num_expert, self.emb_dim,self.mlp_dims, self.dropout = args.mdfend_paras
        self.num_worker = args.num_worker
        self.weight_decay = args.weight_decay
        self.device = device
        self.checkpoint = args.start_epoch
        self.M_lr = args.M_lr
        self.patience = args.patience
        self.dataset = args.dataset
        self.model_path = os.path.join(args.save_dir, args.model_name, args.dataset)

        #=====================================load rumor_detection model================================================

        self.mdfend= MultiDomainFENDModel(self.domain_num, self.num_expert, self.emb_dim,self.mlp_dims, self.dropout)

        self.mdfend.to(self.device)
        print(self.mdfend)

        # =====================================load loss function================================================


        self.bceloss = nn.BCELoss()
        #self.optimizer = torch.optim.Adam([{'stu_text_content':self.stu_text_content.parameters()},
                                          # {'stu_text_style': self.stu_text_style.parameters()},
                                          # {'stu_structure': self.stu_structure.parameters()},
                                          # {'stu_temporal': self.stu_temporal.parameters()},
                                          # {'content_reconduction': self.reconduction.parameters()}
                                          # ], lr= self.lr, betas=(self.b1, self.b2))
        self.optimizer = torch.optim.SGD([{'params': self.mdfend.parameters(),'lr': self.M_lr, 'momentum' : 0.9, 'weight_decay':1e-2}])
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                              lr_lambda=LambdaLR(self.epochs, self.start_epoch,
                                                                               decay_start_epoch= self.weight_decay).step)

        torch.autograd.set_detect_anomaly(True)

    def train(self, datapath, start_epoch, domain_dict):

        nonrumor_loader, rumor_loader = seperate_dataloader(datapath, 'train', self.batch_size, self.num_worker,
                                                            self.num_posts,domain_dict)
        val_loader = normal_dataloader(datapath, 'val', self.batch_size, self.num_worker, self.num_posts,domain_dict)

        # ==================================== train and val dataGAN with model=========================================
        acc_all_check = 0
        loss_all_check = 100
        start_time = time.clock()
        patience = self.patience
        for epoch in range(start_epoch, self.epochs):

            train_loss, acc = self.train_batch(epoch, nonrumor_loader, rumor_loader)
            self.lr_scheduler.step()
            print(train_loss, acc)
            with torch.no_grad():
                val_loss, val_acc = self.eval(val_loader,epoch)
            end_time = time.clock()
            print(val_loss, val_acc)

            if val_loss <= loss_all_check and acc_all_check <= val_acc['acc']:# or
                acc_all_check = val_acc['acc']
                loss_all_check = val_loss
                patience = self.patience
                self.checkpoint = epoch
                self.save(self.model_path, epoch, source=True)

            patience -= 1

            if not patience:
                break

        # ==================================== test mdfend with model================================================
        with torch.no_grad():
            test_loader = normal_dataloader(datapath,'test',self.batch_size, self.num_worker, self.num_posts,domain_dict)

            start_epoch = self.load(self.model_path, self.checkpoint, source= True)

            test_loss,  test_acc = self.eval(test_loader,start_epoch)

            with open(os.path.join(self.model_path, 'predict.txt'), 'a') as f:
               f.write( '\t'.join(list(test_acc.keys())) + '\n' + '\t'.join(map(str,list(test_acc.values()))) + '\n')

    def train_batch(self, epoch, nonrumor_loader, rumor_loader):

        train_loss_value = 0
        acc_value =0
        Mall_ce_loss = 0
        contrative_loss, recovery_loss, domian_loss = 0,0,0

        # if len(rumor_loader) > len(nonrumor_loader):
        #     iterloader = zip(CYCLE(nonrumor_loader), rumor_loader)
        # # elif len(rumor_loader) / len(nonrumor_loader) >= 0.5:
        # #     iterloader = zip(nonrumor_loader, rumor_loader)
        # else:
        #     iterloader = zip(nonrumor_loader, CYCLE(rumor_loader))

        iterloader = zip(nonrumor_loader, rumor_loader)

        for iter, (Nonrumors, Rumors) in enumerate(iterloader):
            xn, yn, An, Dn = Nonrumors
            xr, yr, Ar, Dr = Rumors
            xn = xn.to(self.device)
            yn = yn.to(self.device)
            xr = xr.to(self.device)
            yr = yr.to(self.device)
            Ar = Ar.to(self.device)
            An = An.to(self.device)
            Dr = Dr.to(self.device)
            Dn = Dn.to(self.device)

            x = torch.cat((xn, xr) ,dim=0)
            y = torch.cat((yn, yr), dim=0)
            A = torch.cat((An, Ar), dim=0)
            D = torch.cat((Dn, Dr), dim=0)
            # ====================================train Model============================================
            self.mdfend.train()

            self.optimizer.zero_grad()

            preds = self.mdfend(x, A, D)
            loss = self.bceloss(preds, y)

            loss.backward()
            self.optimizer.step()

            train_loss_value += loss.item()

            pred = torch.where(preds>0.5, 1, 0)
            acc = (pred == y).sum() / len(y)
            acc_value += acc.item()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [model loss: %f] [model acc: %f]"
                % (
                    epoch,
                    self.epochs,
                    iter,
                    len(nonrumor_loader),
                    loss.item(),
                    acc.item()
                )
            )

        train_loss_value = round(train_loss_value /(iter+1),4)
        acc_value = round(acc_value /(iter+1),4)
        return train_loss_value, acc_value


    def evaluation(self, dataloader):

        mean_loss = 0
        Conter_Item = ['TP1', 'FP1', 'FN1', 'TN1', 'TP2', 'FP2', 'FN2', 'TN2']
        Conter = {}
        for item in Conter_Item:
            Conter[item] = 0
        num_sample = 0
        self.mdfend.eval()

        for iter, sample in enumerate(dataloader):
            x, y, A, D = sample
            x = x.to(self.device)
            y_ = y.to(self.device)
            A = A.to(self.device)
            D = D.to(self.device)

            preds = self.mdfend(x, A, D)
            loss = self.bceloss(preds, y)

            loss.backward()
            self.optimizer.step()

            mean_loss += loss.item()

            preds = torch.where(preds > 0.5, 1, 0)

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
        mean_loss = round(mean_loss / (iter + 1), 4)

        return mean_loss, Acc_dict


    def test(self, datapath, dataset, domain_dict):


        test_loader = normal_dataloader(datapath, dataset, self.batch_size, self.num_worker, self.num_posts,domain_dict)
        with torch.no_grad():
            start_epoch = self.load(self.model_path, self.checkpoint, source= False)
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
        self.mdfend.eval()

        for iter, sample in enumerate(dataloader):
            x, y, A, D = sample
            x = x.to(self.device)
            y_ = y.to(self.device)
            A = A.to(self.device)
            D = D.to(self.device)

            preds = self.mdfend(x, A, D)
            preds = torch.where(preds > 0.5, 1, 0)

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
                       'mdfend': self.mdfend.state_dict(),
                       'optimizer': self.optimizer.state_dict(),
                       'checkpoint': epoch}
        torch.save(save_states, os.path.join(model_path, str(epoch) + '_model_states.pkl'))
        print('save classifer : %d epoch' % epoch)



    def load(self, model_path, checkpoint, source=True):
        states_dicts = torch.load( os.path.join(model_path, str(checkpoint) + '_model_states.pkl'))

        self.mdfend.load_state_dict(states_dicts['mdfend'])
        self.optimizer.load_state_dict(states_dicts['optimizer'])
        start_epoch = states_dicts['checkpoint']
        print("load epoch {} success!".format(start_epoch))

        return start_epoch


