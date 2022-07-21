import numpy as np
import torch
from torch import nn
import torch.utils.data as data
import scipy.sparse as sp
import argparse
import time
from tqdm import tqdm
from Mmodel import fism
from dataset import load_data
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1000, help='Seed init.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--dim', type=int, default=64, help='user item size.')
    parser.add_argument('--l_r', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--alpha', type=float, default=0.0005, help='alpha.')
    parser.add_argument('--bata', type=float, default=0.0005, help='bata.')
    parser.add_argument('--lamda', type=float, default=0.0005, help='lamda.')
    parser.add_argument('--gama', type=float, default=0.0005, help='gama.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--topk', type=int, default=6, help='Workers number.')
    parser.add_argument('--save_file', default='result.txt', help='File saving path.')
    parser.add_argument('--n_user', type=int, default=170909, help='user num.')
    parser.add_argument('--m_item', type=int, default=149626, help='item num.')
    parser.add_argument('--negative_num', type=int, default=10, help='negative item num.')
    parser.add_argument('--epoch', type=int, default=40, help='item num.')
    parser.add_argument('--result', type=bool, default=False, help='generate result csv.')
    device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    np.random.seed(seed)
    dim = args.dim
    l_r = args.l_r
    batch_size = args.batch_size  # batch_size

    train_data, test_data, train_mat, n_user, m_item,train_loader,test_loader,mask,test_ground_truth_list,interacted_items,user_item_dict= load_data(batch_size)
    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(args)
    fism = fism(args)
    fism = fism.to(device)
    if args.result==True:
        m_state_dict = torch.load('D:\Desktop\\fism_pytorch\modelpath\\40_epoch.pt')
        fism.load_state_dict(m_state_dict)
        #fism.result(fism, test_loader, mask, args.topk, 40)
    optimizer = torch.optim.Adam(fism.parameters(), lr=l_r)
    train(fism, optimizer, train_loader, test_loader, mask, test_ground_truth_list,interacted_items, args,user_item_dict)
    print('END')