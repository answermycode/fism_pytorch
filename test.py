import numpy as np
import torch
from torch import nn
import torch.utils.data as data
import scipy.sparse as sp
import argparse
import time
from tqdm import tqdm


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k

    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall_n = np.where(recall_n != 0, recall_n, 1)
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}
def MRR(r,k): # 取grandtrue第一个 看在topk中的位置 得分 加起来
    result = []
    score = 1/r
    for i in score:
        if i>0:
            result.append(i)
        else:
            result.append(0)
    return np.sum(np.array(result).astype('float'))



def NDCGatK_r(test_data, r, k):
	"""
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
	assert len(r) == len(test_data)
	pred_data = r[:, :k]

	test_matrix = np.zeros((len(pred_data), k))
	for i, items in enumerate(test_data):
		length = k if k <= len(items) else len(items)
		test_matrix[i, :length] = 1
	max_r = test_matrix
	idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
	dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
	dcg = np.sum(dcg, axis=1)
	idcg[idcg == 0.] = 1.
	ndcg = dcg / idcg
	ndcg[np.isnan(ndcg)] = 0.
	return np.sum(ndcg)



def test_one_batch(X, k):
    sorted_items = X[0].numpy().tolist()
    groundTrue = X[1]
    r,m = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue,r,k),MRR(m,k)

def getLabel(test_data, pred_data):
    r = []
    m = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
        #计算第一个值在topk中的位置
        if groundTrue[0] in predictTopK:
           m.append(predictTopK.index(groundTrue[0])+1)
        else:
            m.append(-1)
    return np.array(r).astype('float'),np.array(m).astype('float')

def test(model, test_loader, test_ground_truth_list, mask, topk, n_user,user_item_dict,interacted_items):
    users_list = []
    rating_list = []
    groundTrue_list = []
    with torch.no_grad():
        model.eval()
        pbar = tqdm(total=len(test_loader))
        for idx, batch_users in enumerate(test_loader):
            user_item = []
            for i in batch_users:
                user_item.append(user_item_dict[i.item()])
            batch_users = batch_users.to(model.get_device())
            user_item_num = torch.tensor(user_item)
            rating = model.test_foward(batch_users,user_item_num,interacted_items)
            rating = rating.cpu()
            rating += mask[batch_users.cpu()].todense()
            _, rating_K = torch.topk(rating, k=topk)
            rating_list.append(rating_K)

            groundTrue_list.append([test_ground_truth_list[u] for u in batch_users])

            pbar.update(1)
        pbar.close()
    X = zip(rating_list, groundTrue_list)
    Recall, Precision, NDCG, MRR = 0, 0, 0, 0

    for i, x in enumerate(X):
        precision, recall, ndcg, mrr = test_one_batch(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg
        MRR += mrr

    Precision /= n_user
    Recall /= n_user
    NDCG /= n_user
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)
    MRR /= n_user

    return F1_score, Precision, Recall, NDCG, MRR