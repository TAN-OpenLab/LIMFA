# -*-coding:utf-8-*-
"""
@Project    : LIMFA
@Time       : 2022/5/16 16:06
@Author     : Danke Wu
@File       : bdann.py
"""
# -*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.nn.functional as F
from model.Encoder import TransformerEncoder

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
class Transformer_Fusion(nn.Module):
    def __init__(self, domain_num, filter_num, f_in, emb_dim,  dropout):
        super(Transformer_Fusion, self).__init__()

        self.domain_num = domain_num

        self.hidden_size = emb_dim
        self.lstm_size = emb_dim

        # self.embed = nn.Linear(f_in, emb_dim, bias=False)
        # # TEXT RNN
        # self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        # self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        # self.text_encoder = nn.Linear(emb_dim, self.hidden_size)

        ### transformer
        self.embedding = nn.Sequential(nn.Linear(f_in, emb_dim, bias=False),
                                       nn.LeakyReLU())
        self.extractor = TransformerEncoder(1, emb_dim, 2, emb_dim, dropout)
        self.grad_reversal = GRL(lambda_=1)

        ## Class  Classifier
        self.class_classifier = nn.Sequential(nn.Linear( self.hidden_size, 2),
                                              nn.Softmax(dim=1))

        ###Event Classifier
        self.domain_classifier = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                               nn.LeakyReLU(True),
                                               nn.Linear(self.hidden_size, self.domain_num))


    def forward(self, text):

        text = self.embedding(text)
        mask_nonzero = torch.nonzero(text.sum(-1), as_tuple=True)
        text = self.extractor(text, mask_nonzero )
        text = torch.mean(text, dim=1)

        ### Fake or real
        class_output = self.class_classifier(text)
        ## Domain (which Event )
        reverse_feature = self.grad_reversal(text)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output, text