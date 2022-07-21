import numpy as np
import torch
from torch import nn
import torch.utils.data as data
import scipy.sparse as sp
import argparse
import time
from tqdm import tqdm

class fism(nn.Module):

    def __init__(self, args):
        super().__init__()
        # user item 隐向量维度
        self.embedding_dim = args.dim
        self.n_user = args.n_user
        self.m_item = args.m_item
        self.bata = args.bata
        self.lamda = args.lamda
        self.alpha = args.alpha
        #初始化bi  qi qu

        '''
        初始化item偏移向量
        '''
        self.bi = nn.Parameter(torch.zeros([self.m_item, 1],dtype=torch.float))
        '''
        初始化物品矩阵,
        '''
        self.qi = nn.Parameter(torch.empty([self.m_item,self.embedding_dim],dtype=torch.float))
        '''
        初始化用户矩阵
        '''
        self.pu = nn.Parameter(torch.empty([self.m_item,self.embedding_dim],dtype=torch.float))
        '''
        初始化用户评分历史矩阵
        '''
        torch.nn.init.normal_(self.qi,mean=0,std=0.001)
        torch.nn.init.normal_(self.pu, mean=0, std=0.001)
    def cal_loss_L(self, users, pos_items, neg_items,user_item_num,interacted_items):
        device = self.get_device()
        user_temp=[]
        for i in users:
            temp = torch.sum(self.pu[torch.tensor(interacted_items[i.item()],dtype=torch.int64).unsqueeze(1)],dim=0)
            user_temp.append(temp)
        user_embeds = torch.cat(user_temp,dim=0)
        b_i_i = self.bi[pos_items]
        b_i_j = self.bi[neg_items]
        bata = self.bata
        lamda = self.lamda
        alpha =self.alpha
        pos_embeds = self.qi[pos_items]
        neg_embeds = self.qi[neg_items]
        num = user_item_num
        t = pow(num,-alpha)
        t = t.to(device)
        pos_scores = t*torch.diag(torch.mm(user_embeds, pos_embeds.t()),0)+b_i_i.t() # batch_size
        neg_scores = t*torch.sum(user_embeds.unsqueeze(1)*neg_embeds,2).t()+torch.sum(b_i_j.transpose(1,2),1).t()  # batch_size * negative_num
        pos_labels = torch.ones(neg_scores.size()).to(device)
        mse = nn.MSELoss(reduction='sum')
        loss = mse(pos_labels,(pos_scores-neg_scores))
        loss = loss + bata * torch.sum(user_embeds**2)+bata*(torch.sum(pos_embeds**2)+torch.sum(neg_embeds**2)) \
               +lamda * (torch.sum(b_i_i**2)+torch.sum(b_i_j**2))
        return loss


    def forward(self,users, pos_items, neg_items,user_item_num,interacted_items):

        loss = self.cal_loss_L(users, pos_items, neg_items,user_item_num,interacted_items)

        return loss

    def test_foward(self, users,user_item_num,interacted_items):
        device = self.get_device()
        items = torch.arange(self.m_item).to(users.device)
        user_temp = []
        for i in users:
            temp = torch.sum(self.pu[torch.tensor(interacted_items[i.item()], dtype=torch.int64).unsqueeze(1)], dim=0)
            user_temp.append(temp)
        user_embeds = torch.cat(user_temp, dim=0)
        item_embeds = self.qi[items]
        bi = self.bi[items]
        num = user_item_num
        t = pow(num, -self.alpha)
        t = t.to(device)
        score = torch.sigmoid(t.unsqueeze(1)*torch.mm(user_embeds, item_embeds.t())+bi.t())
        return score

    def get_device(self):
        return self.pu.device