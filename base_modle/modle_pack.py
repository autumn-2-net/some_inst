import math

import torch
import torch.nn as nn
from base_modle.bert_modle import co_encode_modle,bert_modle
from  base_modle.cov_encode import cov_encode


class main_encoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.img_bert =bert_modle(config=config,layers=config.img_bert_lay)
        self.bh_bert = bert_modle(config=config, layers=config.bh_bert_lay)
        self.co_attentional =co_encode_modle(config=config,layers=config.co_att_lay)

    def forward(self,img,bh,att_mask=None,img_attention_mask=None):
        imgg =self.img_bert(img)
        bhh =self.bh_bert(bh)
        out_img,out_bh = self.co_attentional(imgg,bhh,att_mask)
        return out_img, out_bh

class world_img_embedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embedding=nn.Embedding(num_embeddings=48,embedding_dim=512,padding_idx=0) #词嵌入
        self.position = nn.Embedding(num_embeddings=65,embedding_dim=512,padding_idx=64)  # 1 -64
        self.type = nn.Embedding(num_embeddings=3,embedding_dim=512,padding_idx=2)  # 0-img,1-bh两类
        self.LayerNorm_bh = nn.LayerNorm(config.n_embd, eps=1e-12)
        self.LayerNorm_img = nn.LayerNorm(config.n_embd, eps=1e-12)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x,type):
        seq_length = x.size(1)
        bach_size = x.size(0)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(torch.randn((bach_size,seq_length), device=x.device))
        if type == 0: #img
            token_type_ids = torch.zeros_like(torch.randn((bach_size,seq_length), device=x.device))
            position =self.position(position_ids)
            typee =self.type(token_type_ids)
            return self.LayerNorm_img(x+position+typee)



        if type == 1:
            token_type_ids = torch.ones_like(torch.randn((bach_size,seq_length), device=x.device))
            position = self.position(position_ids)
            typee = self.type(token_type_ids)
            v_embedding = self.embedding(x)
            return self.LayerNorm_bh(v_embedding + position + typee)

class part_of_main_modle(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.cov_encode=cov_encode()
        self.embedding =world_img_embedding(config=config)
        self.main_encoder=main_encoder(config=config)

    def forward(self, img,bh,att_mask=None,att_mask_img=None):
        img_cov =self.cov_encode(img)
        img_emb =self.embedding(img_cov,0)
        bh_embedding =self.embedding(bh,1)

        return self.main_encoder(img_emb,bh_embedding,att_mask)
