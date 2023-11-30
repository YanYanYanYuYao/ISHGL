import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module
import torch.sparse
from numba import jit
import heapq
from tqdm import tqdm

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class HyperConvolution(Module):
    def __init__(self, layers, dataset):
        super(HyperConvolution, self).__init__()
        self.layers = layers
        self.dataset = dataset
    def forward(self, adjacency, embedding):
        item_embeddings = embedding
        item_embedding_layer_input = item_embeddings
        final = [item_embedding_layer_input]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(item_embeddings)
        item_embeddings = np.sum(final, 0) / (self.layers + 1)
        return item_embeddings

class ISHGL(Module):
    def __init__(self, adjacency, n_node, lr, layers, l2, beta, dataset, embedding_size, batch_size):
        super(ISHGL, self).__init__()
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.dataset = dataset
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.adjacency = adjacency
        self.embedding = nn.Embedding(self.n_node, self.embedding_size)
        self.pos_embedding = nn.Embedding(300, self.embedding_size)
        self.HyperGraph = HyperConvolution(self.layers, dataset)
        self.w_1 = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.w_2 = nn.Parameter(torch.Tensor(self.embedding_size, 1))
        self.glu1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.glu2 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_session_embedding(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.embedding_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.embedding_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_embedding = self.pos_embedding.weight[:len]
        pos_embedding = pos_embedding.unsqueeze(0).repeat(self.batch_size, 1, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = self.w_1(torch.cat([pos_embedding, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def forward(self, session_item, session_len, reversed_sess_item, mask):
        item_embedding = self.HyperGraph(self.adjacency, self.embedding.weight)
        session_embedding = self.generate_session_embedding(item_embedding, session_item, session_len, reversed_sess_item, mask)
        return item_embedding, session_embedding, self.beta


@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    return ids

def SSL(session_embedding, target_embedding):
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        return corrupted_embedding
    def score(x1, x2):
        return torch.sum(torch.mul(x1, x2), 1)
    positive = score(session_embedding, target_embedding)
    negative = score(target_embedding, row_shuffle(session_embedding))
    one = torch.cuda.FloatTensor(negative.shape[0]).fill_(1)
    SSL_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(positive)) - torch.log(1e-8 + (one - torch.sigmoid(negative))))
    return SSL_loss



def forward(model, i, data,train):
    tar, session_len, session_item, reversed_session_item, mask = data.get_slice(i)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_session_item).long())
    item_embedding, session_embedding, beta = model(session_item, session_len, reversed_sess_item, mask)
    if train:
        con_loss_1 = SSL(session_embedding,item_embedding[tar])*beta
    else:
        con_loss_1 = 0
    scores = torch.mm(session_embedding, torch.transpose(item_embedding, 1, 0))
    return tar, scores, con_loss_1


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
        i = slices[step]
        model.zero_grad()
        targets, scores, con_loss = forward(model, i, train_data,train=True)
        loss = model.loss_function(scores , targets)
        loss = loss + con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
        i = slices[step]
        tar, scores, con_loss = forward(model, i, test_data,train=False)
        scores = trans_to_cpu(scores).detach().numpy()
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(20, scores[idd]))
        index = np.array(index)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    return metrics, total_loss


