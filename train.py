import numpy as np
import torch
from torch import nn
import torch.utils.data as data
import scipy.sparse as sp
import argparse
import time
from tqdm import tqdm
from test import test

def Sampling(pos_train_data, item_num, neg_ratio, interacted_items, sampling_sift_pos):
    neg_candidates = np.arange(item_num)

    if sampling_sift_pos:
        neg_items = []
        for u in pos_train_data[0]:
            probs = np.ones(item_num)
            probs[interacted_items[u]] = 0
            probs /= np.sum(probs)

            u_neg_items = np.random.choice(neg_candidates, size=neg_ratio, p=probs, replace=True).reshape(1, -1)

            neg_items.append(u_neg_items)

        neg_items = np.concatenate(neg_items, axis=0)
    else:
        neg_items = np.random.choice(neg_candidates, (len(pos_train_data[0]), neg_ratio), replace=True)

    neg_items = torch.from_numpy(neg_items)
    neg_items = neg_items.long()
    # print(pos_train_data,pos_train_data[0],pos_train_data[1])

    return pos_train_data[0], pos_train_data[1], neg_items  # users, pos_items, neg_items

def train(model, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, args,user_item_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
    batches = len(train_loader.dataset) // args.batch_size
    if len(train_loader.dataset) % args.batch_size != 0:
        batches += 1
    print('Total training batches = {}'.format(batches))

    for epoch in tqdm(range(0, args.epoch)):
        model.train()
        start_time = time.time()
        tqm = tqdm(total=batches)
        for batch, x in enumerate(train_loader):
            users, pos_items, neg_items = Sampling(x, args.m_item, args.negative_num, interacted_items, False)
            user_item = []
            for i in users:
                user_item.append(user_item_dict[i.item()])

            user_item_num = torch.tensor(user_item)
            user_item_num.to(device)
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            model.zero_grad()
            loss = model(users, pos_items, neg_items,user_item_num,interacted_items)
            loss.backward()
            optimizer.step()
            tqm.update(1)
        tqm.close()
        train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
        if epoch == 40 and args.result == True:
            torch.save(model.state_dict(), 'D:\Desktop\\fism_pytorch\modelpath\%s_epoch.pt'%epoch)
            model.result(model, test_loader, mask, args.topk, epoch)
        need_test = False
        if epoch % 1 == 0 and args.result == False:
            need_test = True

        if need_test:
            start_time = time.time()
            F1_score, Precision, Recall, NDCG,MRR = test(model, test_loader, test_ground_truth_list, mask, args.topk,
                                                     args.n_user,user_item_dict,interacted_items)
            test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))

            print('The time for epoch {} is: train time = {}, test time = {}'.format(epoch, train_time, test_time))
            print(
                "Loss = {:.5f}, F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}\tMRR: {:.5f}".format(loss.item(),
                                                                                                            F1_score,
                                                                                                            Precision,
                                                                                                            Recall,
                                                                                                            NDCG,MRR))
    print('Training end!')