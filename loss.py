
import torch
from numpy.ma.core import negative
from torch import nn
import torch.nn.functional as F
import itertools
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature",
                             torch.tensor(temperature).cuda())  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).cuda()).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i.squeeze(), dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j.squeeze(), dim=1)  # (bs, dim)  --->  (bs, dim)
        cur_bs = z_i.shape[0]
        if z_i.shape[0] < self.batch_size:
            padded_z_i = torch.full((self.batch_size, z_i.shape[1]), float(0))
            padded_z_i[:cur_bs] = z_i
            padded_z_j = torch.full((self.batch_size, z_j.shape[1]), float(0))
            padded_z_j[:cur_bs] = z_j
            z_i = padded_z_i
            z_j = padded_z_j
        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                                representations.unsqueeze(0),
                                                dim=2).cuda()  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0).cuda()  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(
            similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(
            nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * cur_bs)
        return loss

class MyLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.2, cl_weight=0.2, bce_weight=0.8):
        super().__init__()
        self.cl = ContrastiveLoss2(batch_size, temperature)
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.cl_weight = cl_weight
        self.bce_weight = bce_weight

    def __call__(self, embs, outputs, labels):
        # images, twts, caps, attrs = embs
        images, twts, caps = embs
        cl = 0
        # for emb_i, emb_j in list(itertools.combinations(embs, 2)):
        cl += self.cl(images, twts, labels)
        cl += self.cl(images, caps, labels)
        # cl += self.cl(images, attrs, labels)
        return self.bce_weight * self.bce_loss(outputs, labels) + self.cl_weight * cl



class ContrastiveLoss2(nn.Module):
    def __init__(self, batch_size, temperature=0.2):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature",
                             torch.tensor(temperature).cuda())  # 超参数 温度
        # self.register_buffer("negatives_mask", (
        #     ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).cuda()).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j, labels):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i.squeeze(), dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j.squeeze(), dim=1)  # (bs, dim)  --->  (bs, dim)
        cur_bs = z_i.shape[0]
        # if z_i.shape[0] < self.batch_size:
        #     padded_z_i = torch.full((self.batch_size, z_i.shape[1]), float(0))
        #     padded_z_i[:cur_bs] = z_i
        #     padded_z_j = torch.full((self.batch_size, z_j.shape[1]), float(0))
        #     padded_z_j[:cur_bs] = z_j
        #     z_i = padded_z_i
        #     z_j = padded_z_j
        # representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity = F.cosine_similarity(z_i, z_j, dim=1).cuda()
        positive_indices = torch.where(labels == 1)[0]
        positive_cosine_sim = similarity[positive_indices]
        negative_indices = torch.where(labels == 0)[0]
        negative_cosine_sim = similarity[negative_indices]

        # sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        # sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        # positives = torch.cat([sim_ij, sim_ji], dim=0).cuda()  # 2*bs

        nominator = torch.sum(torch.exp(positive_cosine_sim / self.temperature))
        denominator = torch.sum(torch.exp(
            negative_cosine_sim / self.temperature))  # 2*bs, 2*bs

        loss_partial = torch.log(
            nominator / denominator)  # 2*bs
        loss = loss_partial / cur_bs
        return loss