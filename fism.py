import numpy as np
import torch
from torch import nn
import torch.utils.data as data
import scipy.sparse as sp
import argparse
import time
from tqdm import tqdm
def load_data(batch_size):
    #训练集 测试集
    train_file= "D:\\Desktop\\fism_pytorch\\mangguo\\final_train.npy"
    test_file = "D:\\Desktop\\fism_pytorch\\mangguo\\final_test.npy"

    n_user,m_item = 170909,149626
    train_data = np.load(train_file, allow_pickle=True).tolist()
    test_data = np.load(test_file, allow_pickle=True).tolist()
    #交互矩阵 这里做R 评分矩阵 交互的评分1 未交互的评分0 ##后面考虑用观看时长的比例 评分1到5
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
    indexa = []
    indexb = []
    interacted_items = [[] for _ in range(n_user)]
    for (u, i) in train_data:
        indexa.append(u)
        indexb.append(i)
        #mask[u,i] = -np.inf
        interacted_items[u].append(i)
    mask = sp.csr_matrix(([-np.inf for _ in range(len(train_data))],(indexa,indexb)),shape=(n_user,m_item))

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(n_user)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)

    train_loader = data.DataLoader(list(range(n_user)), batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(list(range(n_user)), batch_size=batch_size, shuffle=False, num_workers=4)
    print("data has been load")


    return train_data, test_data, train_mat, n_user, m_item,train_loader,test_loader,mask,test_ground_truth_list


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
    # print(pos_train_data,pos_train_data[0],pos_train_data[1])

    return pos_train_data[0], pos_train_data[1], neg_items  # users, pos_items, neg_items

def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue,r,k)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

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

def test(model, test_loader, test_ground_truth_list, mask, topk, n_user):
    users_list = []
    rating_list = []
    groundTrue_list = []
    with torch.no_grad():
        model.eval()
        # pbar = tqdm(total=len(test_loader))
        for idx, batch_users in enumerate(test_loader):
            batch_users = batch_users.to(model.get_device())
            rating = model.test_foward(batch_users)
            rating = rating.cpu()

            # rating += mask.index_select(dim=0,index=batch_users).to(model.get_device())
            rating += mask[batch_users.cpu()].todense()
            _, rating_K = torch.topk(rating, k=topk)
            rating_list.append(rating_K)

            groundTrue_list.append([test_ground_truth_list[u] for u in batch_users])

            # pbar.update(1)
        # pbar.close()
    X = zip(rating_list, groundTrue_list)
    Recall, Precision, NDCG = 0, 0, 0

    for i, x in enumerate(X):
        precision, recall, ndcg = test_one_batch(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg

    Precision /= n_user
    Recall /= n_user
    NDCG /= n_user
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    return F1_score, Precision, Recall, NDCG

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
        #初始化bi bu qi qu yi
        self.bi = nn.Parameter(torch.zeros([1, self.m_item],dtype=torch.float))
        '''
        初始化用户向量
        '''
        self.bu = nn.Parameter(torch.zeros([1, self.n_user],dtype=torch.float))
        '''
        初始化物品矩阵,这里提前转制
        '''
        self.qi = nn.Parameter(torch.randn([self.m_item,self.embedding_dim],dtype=torch.float))
        '''
        初始化用户矩阵
        '''
        self.pu = nn.Parameter(torch.randn([self.n_user,self.embedding_dim],dtype=torch.float))
        '''
        初始化用户评分历史矩阵
        '''

    def cla_loss(self, users, pos_items, neg_items):
        device = self.get_device()
        b_i_i = self.bi(pos_items)
        b_i_j = self.bi(neg_items)
        bata = self.bata
        lamda = self.lamda
        alpha =self.alpha
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)
        num = pos_items.shape(0)-1
        t = pow(num,-alpha)
        pos_scores = t*(user_embeds * pos_embeds).sum(dim=-1)+b_i_i  # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = t*(user_embeds * neg_embeds).sum(dim=-1)+b_i_j  # batch_size * negative_num
        neg_labels = torch.zeros(neg_scores.size()).to(device)

        pos_labels = torch.ones(pos_scores.size()).to(device)

        loss = nn.MSELoss(pos_labels,(pos_scores-neg_scores))
        loss = loss.sum() + bata * torch.sum(user_embeds**2)+bata*(torch.sum(pos_embeds**2)+torch.sum(neg_embeds**2)) \
               +lamda * (torch.sum(b_i_i**2)+torch.sum(b_i_j**2))
        return loss


    def forward(self,users, pos_items, neg_items):

        loss = self.cal_loss_L(users, pos_items, neg_items)

        return loss

    def test_foward(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)
        bi = self.bi(items)

        return user_embeds.mm(item_embeds.t())+bi
    def get_device(self):
        return self.user_embeds.weight.device


def train(model, optimizer, train_loader, test_loader, mask, test_ground_truth_list,interacted_items, args):
    device = args.device
    batches = len(train_loader.dataset) // args.batch_size
    if len(train_loader.dataset) % args.batch_size != 0:
        batches += 1
    print('Total training batches = {}'.format(batches))


    for epoch in tqdm(range(0,args.epoch)):
        model.train()
        start_time = time.time()
        for batch, x in enumerate(train_loader):
            users, pos_items, neg_items = Sampling(x, args.n_item, args.negative_num, interacted_items,False)
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            model.zero_grad()
            loss = model(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
        train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
        need_test = False
        if epoch % 10 == 0:
            need_test = True

        if need_test:
            start_time = time.time()
            F1_score, Precision, Recall, NDCG = test(model, test_loader, test_ground_truth_list, mask, args.topk,
                                                     args.m_user)
            test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))

            print('The time for epoch {} is: train time = {}, test time = {}'.format(epoch, train_time, test_time))
            print("Loss = {:.5f}, F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(loss.item(),
                                                                                                            F1_score,
                                                                                                            Precision,
                                                                                                            Recall,
                                                                                                            NDCG))
    print('Training end!')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1000, help='Seed init.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--dim', type=int, default=64, help='user item size.')
    parser.add_argument('--l_r', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--alpha', type=float, default=0.0005, help='alpha.')
    parser.add_argument('--bata', type=float, default=0.0005, help='bata.')
    parser.add_argument('--lamda', type=float, default=0.0005, help='lamda.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--topK', type=int, default=10, help='Workers number.')
    parser.add_argument('--save_file', default='result.txt', help='File saving path.')
    parser.add_argument('--n_user', type=int, default=170909, help='user num.')
    parser.add_argument('--m_item', type=int, default=149606, help='item num.')

    device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    dim = args.dim
    l_r = args.l_r
    batch_size = args.batch_size  # batch_size

    train_data, test_data, train_mat, n_user, m_item,train_loader,test_loader,mask,test_ground_truth_list= load_data(batch_size)
    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(args)
    ultragcn = fism(args)
    ultragcn = ultragcn.to(args.device)
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=l_r)
    train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, args)
    print('END')



























