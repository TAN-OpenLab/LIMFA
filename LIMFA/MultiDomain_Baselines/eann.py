# -*-coding:utf-8-*-
"""
@Project    : LIMFA
@Time       : 2022/5/12 10:08
@Author     : Danke Wu
@File       : eann.py
"""

import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.nn.functional as F

class Grl_func(Function):
    def __init__(self):
        super(Grl_func, self).__init__()

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return -lambda_ * grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_=1.):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return Grl_func.apply(x, self.lambda_)


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, domain_num, filter_num, f_in, emb_dim,  dropout):
        super(CNN_Fusion, self).__init__()

        self.domain_num = domain_num

        self.hidden_size = emb_dim
        self.lstm_size = emb_dim

        # self.embed = nn.Linear(f_in, emb_dim, bias=False)
        # # TEXT RNN
        # self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        # self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        # self.text_encoder = nn.Linear(emb_dim, self.hidden_size)

        ### TEXT CNN
        channel_in = 1
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, f_in)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.grad_reversal = GRL(lambda_=1)

        ## Class  Classifier
        self.class_classifier = nn.Sequential(nn.Linear( self.hidden_size, 2),
                                              nn.Softmax(dim=1))

        ###Event Classifier
        self.domain_classifier = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                               nn.LeakyReLU(True),
                                               nn.Linear(self.hidden_size, self.domain_num))


    def forward(self, text):

        ##########CNN##################
        text = text.unsqueeze(dim=1)# add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]
        # text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        text = F.leaky_relu(self.fc1(text))

        ### Fake or real
        class_output = self.class_classifier(text)
        ## Domain (which Event )
        reverse_feature = self.grad_reversal(text)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output, text