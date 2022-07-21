import numpy as np
import torch
from torch import nn
import pandas as pd
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
        self.gama = args.gama
        #初始化bi  qi qu

        '''
        初始化item偏移向量
        '''
        self.bu = nn.Parameter(torch.zeros([self.n_user, 1],dtype=torch.float))
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
        torch.nn.init.normal_(self.qi, mean=0, std=0.001)
        torch.nn.init.normal_(self.pu, mean=0, std=0.001)

    def cal_loss_L(self, users, pos_items, neg_items,user_item_num,interacted_items):
        device = self.get_device()
        user_temp=[]
        for i in users:
            temp = torch.sum(self.pu[torch.tensor(interacted_items[i.item()],dtype=torch.int64).unsqueeze(1)],dim=0)
            user_temp.append(temp)
        user_embeds = torch.cat(user_temp,dim=0)
        b_i = self.bi[pos_items]
        b_j = self.bi[neg_items]
        b_u = self.bu[users]
        bata = self.bata
        lamda = self.lamda
        alpha =self.alpha
        pos_embeds = self.qi[pos_items]
        neg_embeds = self.qi[neg_items]
        num = user_item_num
        t = pow(num,-alpha)
        t = t.to(device)
        pos_scores = torch.sigmoid(t*torch.diag(torch.mm(user_embeds, pos_embeds.t()),0)+b_i.t()+b_u.t())# batch_size
        neg_scores = torch.sigmoid(t*torch.sum(user_embeds.unsqueeze(1)*neg_embeds,2).t()+torch.sum(b_j.transpose(1,2),1).t()+b_u.t())
        pos_labels = torch.ones(pos_scores.size()).to(device)
        neg_labels = torch.zeros(neg_scores.size()).to(device)
        mse = nn.MSELoss(reduction='sum')
        loss_pos = mse(pos_labels,pos_scores)
        loss_neg = mse(neg_labels,neg_scores)
        loss = loss_pos+loss_neg + bata * torch.sum(user_embeds**2)+bata*torch.sum(pos_embeds**2) \
               +lamda * (torch.sum(b_i**2)+self.gama*torch.sum(b_u**2))
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
        bu = self.bu[users]
        num = user_item_num
        t = pow(num, -self.alpha)
        t = t.to(device)
        score = torch.sigmoid(t.unsqueeze(1)*torch.mm(user_embeds, item_embeds.t())+bi.t()+bu)
        return score

    def get_device(self):
        return self.pu.device

    def result(self, model, test_loader, mask, topk, epoch):
        rating_list = []
        candi_item_id = np.load("D:\Desktop\\fism_pytorch\mangguo\candi_item_id.npy", allow_pickle=True).tolist()
        with torch.no_grad():
            model.eval()
            # pbar = tqdm(total=len(test_loader))
            for idx, batch_users in enumerate(test_loader):
                filt_candi = torch.full((batch_users.size(0), 149626), -np.inf)
                for i in candi_item_id:
                    filt_candi[:, i] = 0
                batch_users = batch_users.to(model.get_device())
                rating = model.test_foward(batch_users)
                rating = rating.cpu()
                rating += filt_candi
                rating += mask[batch_users]
                _, rating_K = torch.topk(rating, k=topk)
                rating_list.append(rating_K)
                # pbar.update(1)
            # pbar.close()
        result = {}
        id = 0
        for i in rating_list:
            for j in i:
                result[id] = j.tolist()
                id = id + 1

        id2did = np.load("D:\Desktop\\fism_pytorch\mangguo\id2did_dict.npy", allow_pickle=True).item()
        item2id = np.load("D:\Desktop\fism_pytorch\mangguo\id2item_dict.npy", allow_pickle=True).item()
        id2item = {v: k for k, v in item2id.items()}
        list_result = []
        for i in result:
            num = 0
            for j in result[i]:
                num = num + 1
                list_result.append([id2did[str(i)], id2item[j], num])

        final = pd.DataFrame(list_result, columns=['did', 'vid', 'rank'])
        final.to_csv("/home/qk/code/UltraGCN-main/result/resul_new%s.csv" % epoch, index=None)
        print("结果已生成")
