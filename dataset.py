import numpy as np
import torch
from torch import nn
import torch.utils.data as data
import scipy.sparse as sp
import argparse
import time
from tqdm import tqdm

# class mydataset(data.Dataset):
#     def __init__(self,train_data):
#         items = []
#         dict_items = {}
#         for x in train_data:
#             dict_items[x[0]] = []
#         for x in train_data:
#             dict_items[x[0]].append(x[1] + 1)
#         for i in dict_items:
#             items.append([i,dict_items[i]])
#         self.user_item = items
#
#     def __getitem__(self, index):
#         user = self.user_item[index][0]
#         item = self.user_item[index][1]
#         return user, item
#     def __len__(self):
#         return len(self.user_item)
# def collate_fn(batch):
#     batch = list(zip(*batch))
#     user = list(batch[0])
#     item = list(batch[1])
#     del batch
#     return user, item
def load_data(batch_size):
    # 训练集 测试集
    train_file = "D:\\Desktop\\fism_pytorch\\final_train.npy"
    test_file = "D:\\Desktop\\fism_pytorch\\final_test.npy"

    n_user, m_item = 170909, 149626
    train_data = np.load(train_file, allow_pickle=True).tolist()
    test_data = np.load(test_file, allow_pickle=True).tolist()
    # 交互矩阵 这里做R 评分矩阵 交互的评分1 未交互的评分0 ##后面考虑用观看时长的比例 评分1到5
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
    indexa = []
    indexb = []
    interacted = [[] for _ in range(n_user)]
    for (u, i) in train_data:
        indexa.append(u)
        indexb.append(i)
        # mask[u,i] = -np.inf
        interacted[u].append(i)
    mask = sp.csr_matrix(([-np.inf for _ in range(len(train_data))], (indexa, indexb)), shape=(n_user, m_item))
    interacted_items=[]
    for i in interacted:
        interacted_items.append([i[-1]])

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(n_user)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)
    #统计每个user交互的数量
    user_item_num={}
    for index,i in enumerate(interacted_items):
        user_item_num[index]=len(i)

    #dataset = mydataset(train_data)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = data.DataLoader(list(range(n_user)), batch_size=batch_size, shuffle=False, num_workers=1)
    print("data has been load")

    return train_data, test_data, train_mat, n_user, m_item, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, user_item_num

if __name__ == '__main__':
    train_data, test_data, train_mat, n_user, m_item, train_loader, test_loader, mask, test_ground_truth_list, interacted_items,user_item_dict = load_data(1024)
    for user,item in enumerate(train_loader):
        print(user)
    print(1)
