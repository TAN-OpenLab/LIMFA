# -*-coding:utf-8-*-
"""
@Project    : LIMFA
@Time       : 2022/5/2 20:24
@Author     : Danke Wu
@File       : Model_layers.py
"""
# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Encoder import TransformerEncoder,Decoder
import math
import copy
import torch_scatter


class sentence_embedding(nn.Module):
    def __init__(self, h_in, h_out):
        super(sentence_embedding, self).__init__()
        self.embedding = nn.Linear(h_in, h_out)
        self.leakrelu = nn.LeakyReLU()
        # self.insnorm = nn.InstanceNorm1d(num_posts,affine=False)

    def forward(self,x, mask_nonzero):

        x = self.embedding(x)
        mask_nonzero_matrix = torch.clone(x)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        x = x - mask_nonzero_matrix.detach()
        x = self.leakrelu(x)
        return x


class Extractor(nn.Module):
    def __init__(self, num_layers, n_head, h_in, h_hid, dropout):
        super(Extractor,self).__init__()
        self.encoder = TransformerEncoder(num_layers, h_in, n_head,  h_hid, dropout)
        self.post_attn = Post_Attn(h_in)

    def forward(self,x,  mask_nonzero):

        xf = self.encoder(x,mask_nonzero)
        xf = torch.sum(xf, dim=1) / torch.sum(x.sum(-1) != 0, dim=1, keepdim=True)
        # xf,attn = self.post_attn(xf,mask_nonzero)

        return xf

class Content_reconstruction(nn.Module):
    def __init__(self,num_layers, n_posts,  h_hid, n_head, e_hid, c_in, dropout, pos_embed=None):
        super(Content_reconstruction, self).__init__()
        self.n_posts = n_posts
        self.attn_inverse = nn.Parameter(torch.FloatTensor(n_posts,1))
        self.decoder = Decoder(num_layers, h_hid, n_head, e_hid,  dropout, pos_embed)
        self.fc = nn.Linear(h_hid, c_in)
        nn.init.kaiming_uniform_(self.attn_inverse)

    def forward(self, xc, xs, mask_nonzero):

        # x = torch.cat((xc,xs), dim=-1)
        x = xc + xs
        x = x.unsqueeze(dim=1)
        x = torch.repeat_interleave(x, self.n_posts, dim=1)
        mask_nonzero_matrix = torch.clone(x)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        x = x - mask_nonzero_matrix.detach()

        # attn_ = torch.mul(attn, self.attn_inverse)
        x = torch.mul(self.attn_inverse, x)

        recov_xc = self.decoder(x, mask_nonzero)
        recov_xc = self.fc(recov_xc)
        mask_nonzero_matrix = torch.clone(recov_xc)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        recov_xc = recov_xc - mask_nonzero_matrix.detach()

        return recov_xc


class Post_Attn(nn.Module):
    def __init__(self,h_in):
        super(Post_Attn, self).__init__()
        self.Attn = nn.Linear(2 * h_in,1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask_nonzero):
        (batch, row) = mask_nonzero
        root = torch.zeros_like(x,device=x.device)
        root[batch,row,:] = x[batch,0,:]

        x_plus = torch.cat([x,root],dim=-1)
        attn = self.Attn(x_plus)
        attn.masked_fill_(attn ==0, -1e20)
        attn = self.softmax(attn)
        x = torch.matmul(x.permute(0, 2, 1),attn).squeeze()
        return x, attn


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class LIMFA(nn.Module):
    def __init__(self, c_in, num_layers, n_head, c_hid, e_hid, dropout, num_posts):
        super(LIMFA, self).__init__()

        self.embedding = sentence_embedding( c_in, c_hid)
        self.extractor = Extractor(num_layers,n_head, c_hid, e_hid, dropout)
        self.reconstruction = Content_reconstruction(num_layers, num_posts, c_hid, n_head,
                                                     e_hid, c_in, dropout)

        self.domaingate = nn.Sequential(nn.Linear(c_hid,c_hid),
                                        nn.Sigmoid())

        self.classifier_all = nn.Linear(c_hid, 2)
        self.classifier_com = nn.Linear( c_hid, 2)

    def gateFeature(self,x, gate):

        xc = torch.mul(x, gate)
        xs = torch.mul(x, 1 - gate)

        return xc, xs

    def forward(self, x):

        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
        x = self.embedding(x, mask_nonzero)

        xf = self.extractor(x, mask_nonzero)  # , stance
        # xs = self.extractor2(x, A, mask_nonzero)
        dgate = self.domaingate(xf)
        xc, xs = self.gateFeature(xf, dgate)

        xall = xc + xs
        preds = self.classifier_all(xall)
        preds_xs = self.classifier_all(xc.detach() + xs)

        preds_xc = self.classifier_com(xc)

        x_rec = self.reconstruction(xc, xs, mask_nonzero)

        return preds, preds_xc,preds_xs, xc, xs, x_rec, dgate
