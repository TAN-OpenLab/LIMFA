# -*-coding:utf-8-*-
"""
@Project    : LIMFA
@Time       : 2022/4/22 9:34
@Author     : Danke Wu
@File       : LIMFA_NET.py
"""
# -*-coding:utf-8-*-

import random
import torch
import torch.autograd as autograd
import torch.nn as nn
from dataset.Dataloader_enhance import seperate_dataloader, normal_dataloader
from Loss_Functions.evaluation import *
from Loss_Functions.loss_wrapper import LossWrapper_center
from model.Model_layers import sentence_embedding, LambdaLR, LIMFA
import os, sys
import time
from itertools import cycle as CYCLE


class LIMFA_Net(object):
    def __init__(self, args, device):
        # parameters

        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.batch_size = args.batch_size
        self.num_posts = args.num_posts
        self.syn_layer = args.data_syn_layer
        self.c_in,self.c_hid = args.text_embedding
        self.num_layers, self.c_hid, self.n_head,self.e_hid, self.dropout = args.encoder_pars
        self.num_worker = args.num_worker
        self.lr = args.lr
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay
        self.device = device
        self.b1, self.b2 = args.b1, args.b2
        self.checkpoint = args.start_epoch
        self.patience = args.patience
        self.dataset = args.dataset
        self.model_path = os.path.join(args.save_dir, args.model_name, args.dataset)

        #=====================================load rumor_detection model================================================

        self.model= LIMFA(self.c_in, self.num_layers, self.n_head, self.c_hid, self.e_hid, self.dropout, self.num_posts)
        # self.r_anchor, self.n_anchor = torch.zeros(1, self.c_hid, requires_grad=False).to(self.device), -torch.zeros(
        #     1, self.c_hid, requires_grad=False).to(self.device)


        self.model.to(self.device)
        print(self.model)

        # =====================================load loss function================================================
        self.ce = nn.CrossEntropyLoss()
        self.mseloss = nn.MSELoss()

        self.loss_wrapper = LossWrapper_center()
        self.loss_wrapper.to(self.device)
        self.optimizer = torch.optim.SGD([{'params': self.model.parameters(),'lr': self.lr, 'momentum' : 0.9, 'weight_decay':1e-4}])
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                              lr_lambda=LambdaLR(self.epochs, self.start_epoch,
                                                                               decay_start_epoch= self.weight_decay).step)

        torch.autograd.set_detect_anomaly(True)

    def train_epoch(self, datapath, start_epoch):

        nonrumor_loader, rumor_loader = seperate_dataloader(datapath, 'train', self.batch_size, self.num_worker,
                                                            self.num_posts)
        val_loader = normal_dataloader(datapath, 'val', self.batch_size, self.num_worker, self.num_posts)

        # ==================================== train and val dataGAN with model=========================================
        acc_all_check = 0
        loss_all_check = 100
        loss_ce_check = 100
        acc_trans_check = 0
        cent_loss_check = 100
        loss_trans_check = {}
        start_time = time.clock()
        patience = self.patience
        for epoch in range(start_epoch, self.epochs):

            train_loss, all_celoss, xc_celoss, cent_loss, recoverey_loss, acc_dict = self.train_batch(epoch,
                                                                                                                nonrumor_loader,
                                                                                                                rumor_loader)
            self.lr_scheduler.step()
            print(train_loss, all_celoss, xc_celoss, cent_loss, recoverey_loss)
            with torch.no_grad():
                val_loss, all_celoss, xc_celoss, cent_loss, recoverey_loss, val_acc_dict = self.evaluation(val_loader,
                                                                                                           epoch)
            end_time = time.clock()
            print(val_loss, all_celoss, xc_celoss, cent_loss, recoverey_loss, val_acc_dict['model_DI']['Acc_all'])

            if acc_all_check < val_acc_dict['model']['Acc_all'] and loss_all_check > all_celoss:  # or
                acc_all_check = val_acc_dict['model']['Acc_all']
                loss_all_check = all_celoss
                self.save(self.model_path, epoch, source=True)

            if acc_trans_check < val_acc_dict['model_DI']['Acc_all'] and  loss_ce_check > xc_celoss +all_celoss:
                loss_ce_check = xc_celoss+all_celoss
                acc_trans_check = val_acc_dict['model_DI']['Acc_all']
                loss_trans_check = val_loss
                self.save(self.model_path, epoch, source=False)
                self.checkpoint = epoch
                patience = self.patience

            patience -= 1

            if not patience:
                break

        # ==================================== test LIMFA with model================================================
        with torch.no_grad():
            test_loader = normal_dataloader(datapath, 'test',self.batch_size, self.num_worker, self.num_posts)

            start_epoch = self.load(self.model_path, self.checkpoint, source= False)

            test_loss, all_celoss, xc_celoss, cent_loss, recoverey_loss,  test_acc_dict = self.evaluation(test_loader,start_epoch)

            with open(os.path.join(self.model_path, 'predict.txt'), 'a') as f:
               f.write( '\t'.join(list(test_acc_dict.keys())) + '\n' + '\t'.join(map(str,list(test_acc_dict.values()))) + '\n')

    def train_batch(self, epoch, nonrumor_loader, rumor_loader):

        train_loss_value = 0
        acc_value =0
        Mall_ce_loss, XC_ce_loss, Center_loss, Recovery_loss = 0, 0, 0, 0
        # r_anchor, n_anchor = torch.zeros(1, self.c_hid, requires_grad=False).to(self.device), torch.zeros(
        #         1, self.c_hid, requires_grad=False).to(self.device)

        iterloader = zip(nonrumor_loader, rumor_loader)

        for iter, (Nonrumors, Rumors) in enumerate(iterloader):
            xn, yn, An = Nonrumors
            xr, yr, Ar = Rumors
            xn = xn.to(self.device)
            yn = yn.to(self.device)
            xr = xr.to(self.device)
            yr = yr.to(self.device)
            Ar = Ar.to(self.device)
            An = An.to(self.device)

            x = torch.cat((xn, xr) ,dim=0)
            y = torch.cat((yn, yr), dim=0)
            A = torch.cat((An, Ar), dim=0)
            # ====================================train Model============================================
            self.model.train()

            self.optimizer.zero_grad()

            # r_anchor_temp, n_anchor_temp = self.r_anchor / (epoch+1), self.n_anchor/ (epoch+1)
            # [r_anchor_temp, n_anchor_temp], r_anchor_temp, n_anchor_temp
            preds, preds_xc, preds_xs, xc, xs, x_rec, fgate = self.model(x)
            loss, (all_celoss, xc_celoss, cent_loss, rec_loss ) = self.loss_wrapper([preds, preds_xc, preds_xs, xc, xs, x_rec, fgate], [y,x])
            # r_anchor += r_anchor_temp.data
            # n_anchor += n_anchor_temp.data

            loss.backward()
            self.optimizer.step()

            train_loss_value += loss.item()
            Mall_ce_loss += all_celoss
            XC_ce_loss += xc_celoss
            Center_loss += cent_loss
            Recovery_loss += rec_loss
            pred = preds.data.max(1)[1]
            acc = (pred == y).sum() / len(y)
            acc_value += acc.item()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [model loss: %f] [model acc: %f] [model ce_loss : %f] [center loss : %f][recovery loss : %f]"
                % (
                    epoch,
                    self.epochs,
                    iter,
                    len(nonrumor_loader),
                    loss.item(),
                    acc.item(),
                    all_celoss,
                    cent_loss,
                    rec_loss
                )
            )

        train_loss_value = round(train_loss_value /(iter+1),4)
        Mall_ce_loss = round( Mall_ce_loss /(iter+1),4)
        Center_loss = round( Center_loss /(iter+1),4)
        recovery_loss = round( Recovery_loss / (iter + 1), 4)
        XC_ce_loss = round(XC_ce_loss/ (iter + 1), 4)
        acc_value = round(acc_value /(iter+1),4)
        # self.r_anchor += r_anchor / (iter+1)
        # self.n_anchor += n_anchor / (iter+1)
        return train_loss_value, Mall_ce_loss, XC_ce_loss, Center_loss, recovery_loss, acc_value


    def evaluation(self, dataloader, epoch):

        self.model.eval()

        mean_loss =0
        Mall_ce_loss, XC_ce_loss, Center_loss, Recovery_loss = 0, 0, 0, 0
        num_sample = 0
        Conter = {}
        ACC_dict = {}
        Conter_Item = ['TP1', 'FP1', 'FN1', 'TN1', 'TP2', 'FP2', 'FN2', 'TN2']
        models = ['model', 'model_DI']
        for model in models:
            Conter[model] = {}
            ACC_dict[model] ={}
            for item in Conter_Item:
                Conter[model][item] = 0

        for iter, sample in enumerate(dataloader):
            x, y, A = sample
            x = x.to(self.device)
            A = A.to(self.device)
            y = y.to(self.device)
            num_sample += len(y)

            # r_anchor, n_anchor = self.r_anchor, self.n_anchor, , r_anchor, n_anchor,[r_anchor, n_anchor]
            preds, preds_xc, preds_xs, xc, xs, x_rec, fgate= self.model(x)
            loss, (all_celoss, xc_celoss, cent_loss, rec_loss) = self.loss_wrapper([preds, preds_xc,preds_xs, xc, xs, x_rec, fgate],
                                                                         [y,x])
            mean_loss += loss.item()
            Mall_ce_loss += all_celoss
            XC_ce_loss += xc_celoss
            Center_loss += cent_loss
            Recovery_loss += rec_loss

            preds_xc = preds_xc.data.max(1)[1].cpu()
            preds = preds.data.max(1)[1].cpu()

            for model in models:
                if model == 'model':
                    (tp1, fn1, fp1, tn1, tp2, fn2, fp2, tn2) = count_2class(preds, y.data.cpu())

                else:
                    (tp1, fn1, fp1, tn1, tp2, fn2, fp2, tn2) = count_2class(preds_xc, y.data.cpu())
                Conter[model]['TP1'] += tp1
                Conter[model]['FN1'] += fn1
                Conter[model]['FP1'] += fp1
                Conter[model]['TN1'] += tn1
                Conter[model]['TP2'] += tp2
                Conter[model]['FN2'] += fn2
                Conter[model]['FP2'] += fp2
                Conter[model]['TN2'] += tn2


        for model in models:
            ACC_dict[model] = evaluationclass(Conter[model], num_sample)
            print(model, ACC_dict[model].items())

        mean_loss = round(mean_loss/(iter+1),4)
        Mall_ce_loss = round(Mall_ce_loss/(iter+1),4)
        Center_loss = round(Center_loss / (iter + 1), 4)
        Recovery_loss= round(Recovery_loss / (iter + 1), 4)
        XC_ce_loss = round(XC_ce_loss / (iter + 1), 4)

        return mean_loss, Mall_ce_loss, XC_ce_loss, Center_loss,Recovery_loss, ACC_dict


    def test(self, datapath, dataset):

        test_loader = normal_dataloader(datapath, dataset, self.batch_size, self.num_worker, self.num_posts)
        with torch.no_grad():
            start_epoch = self.load(self.model_path, self.checkpoint, source=False)
            acc_test_dict = self.content_test(test_loader)
            with open(os.path.join(self.model_path, 'predict.txt'), 'a') as f:
                for model in acc_test_dict.keys():
                    f.write('target doamin' + '\t'+ model+ '\t' + str(start_epoch) + '\n' +
                            '\t'.join(list(acc_test_dict[model].keys())) + '\n' + '\t'.join(
                        map(str, list(acc_test_dict[model].values()))) + '\n')

        return 0


    def content_test(self, dataloader):

        self.model.eval()
        num_sample = 0
        Conter = {}
        ACC_dict = {}
        Conter_Item = ['TP1', 'FP1', 'FN1', 'TN1', 'TP2', 'FP2', 'FN2', 'TN2']
        models = ['model', 'model_DI']
        for model in models:
            Conter[model] = {}
            ACC_dict[model] = {}
            for item in Conter_Item:
                Conter[model][item] = 0

        for iter, sample in enumerate(dataloader):
            x, y, A = sample
            x = x.to(self.device)
            A = A.to(self.device)
            num_sample += len(y)

            preds, preds_xc, preds_xs, xc, xs, x_rec, fgate = self.model(x)
            preds_xc = preds_xc.data.max(1)[1].cpu()
            preds = preds.data.max(1)[1].cpu()

            for model in models:
                if model == 'model':
                    (tp1, fn1, fp1, tn1, tp2, fn2, fp2, tn2) = count_2class(preds, y.data.cpu())

                else:
                    (tp1, fn1, fp1, tn1, tp2, fn2, fp2, tn2) = count_2class(preds_xc, y.data.cpu())
                Conter[model]['TP1'] += tp1
                Conter[model]['FN1'] += fn1
                Conter[model]['FP1'] += fp1
                Conter[model]['TN1'] += tn1
                Conter[model]['TP2'] += tp2
                Conter[model]['FN2'] += fn2
                Conter[model]['FP2'] += fp2
                Conter[model]['TN2'] += tn2

        for model in models:
            ACC_dict[model] = evaluationclass(Conter[model], num_sample)
            print(model, ACC_dict[model].items())

        return ACC_dict

    def save(self, model_path, epoch, source = True):
        save_states = {'model': self.model.state_dict(),
                       'optimizer': self.optimizer.state_dict(),
                       'checkpoint': epoch,
                       # 'rumor_anchor': self.r_anchor,
                       # 'nonrumor_anchor': self.n_anchor

        }
        if source:
            torch.save(save_states, os.path.join(model_path, str(epoch) + '_model_states.pkl'))
            print('save classifer : %d epoch' % epoch)
        else:
            torch.save(save_states, os.path.join(model_path, str(epoch) + '_transfer_states.pkl'))
            print('save transfer_model : %d epoch' % epoch)

    def load(self, model_path, checkpoint, source= True):
        if source:
            states_dicts = torch.load( os.path.join(model_path, str(checkpoint) + '_model_states.pkl'))
        else:
            states_dicts = torch.load( os.path.join(model_path, str(checkpoint) + '_transfer_states.pkl'))

        self.model.load_state_dict(states_dicts['model'])
        self.optimizer.load_state_dict(states_dicts['optimizer'])
        start_epoch = states_dicts['checkpoint']
        # self.r_anchor = states_dicts['rumor_anchor']
        # self.n_anchor = states_dicts['nonrumor_anchor']
        print("load epoch {} success!".format(start_epoch))

        return start_epoch


