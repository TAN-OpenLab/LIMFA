# -*-coding:utf-8-*-
"""
@Project    : LIMFA
@Time       : 2022/5/10 21:23
@Author     : Danke Wu
@File       : mdfend.py
"""
import torch
import torch.nn as nn
from MultiDomain_Baselines.layers import cnn_extractor, MaskAttention, SelfAttentionFeatureExtract, MLP

class MultiDomainFENDModel(torch.nn.Module):
    def __init__(self, domain_num, num_expert,emb_dim, mlp_dims, dropout):
        super(MultiDomainFENDModel, self).__init__()
        self.domain_num = domain_num
        self.gamma = 10
        self.num_expert = num_expert
        self.fea_size = 256

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(nn.Linear(emb_dim * 2, mlp_dims[-1]),
                                  nn.ReLU(),
                                  nn.Linear(mlp_dims[-1], self.num_expert),
                                  nn.Softmax(dim=1))

        self.attention = MaskAttention(emb_dim)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.specific_extractor = SelfAttentionFeatureExtract(multi_head_num=1, input_size=emb_dim,
                                                              output_size=self.fea_size)
        self.classifier = MLP(320, mlp_dims, dropout)

    def forward(self, init_feature, A, idxs):

        masks = torch.where(init_feature.sum(dim=-1)==0, 0, 1)

        feature, _ = self.attention(init_feature, masks)
        # idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)

        gate_input_feature = feature
        gate_input = torch.cat([domain_embedding, gate_input_feature], dim=-1)
        gate_value = self.gate(gate_input)

        shared_feature = 0
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))

        label_pred = self.classifier(shared_feature)

        return torch.sigmoid(label_pred.squeeze(1))