
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss
eps = 1e-5
T1 = 0.2  # 0.05

class SELFContrastiveLoss(nn.Module):
    def __init__(self, batch_size,temperature,low_feature_dim,view_num, device='cuda:0'):
        super(SELFContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size, batch_size, dtype=torch.bool).to(device)).float())
        # self.temperature=temperature#####################################################################
        
        
        self.device=device
        self.low_feature_dim=low_feature_dim
        
        
        ########################################################
        self.similarity = nn.CosineSimilarity(dim=2)
        # self.class_num=number_class
        # self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.criterion = L1Loss()
        ####################################################
        self.classifier3_sample=nn.Linear(low_feature_dim*(view_num),1)
        self.classifier3_cluster=nn.Linear(low_feature_dim*(view_num),1)
    def forward(self, q, k):
        q=q.to(self.device)
        k=k.to(self.device)
        q = F.normalize(q, dim=1)  # (bs, dim)  --->  (bs, dim)
        k = F.normalize(k, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([q, k], dim=0)  # (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                                representations.unsqueeze(0), dim=2)  # (2*bs, 2*bs)
        sim_qk = torch.diag(similarity_matrix, self.batch_size)  # (bs,)
        sim_kq = torch.diag(similarity_matrix, -self.batch_size)  # (bs,)

        nominator_qk = torch.exp(sim_qk / self.temperature)   # (bs,)
        # print(f"similarity_matrix.shape: {similarity_matrix.shape}")

        negatives_qk = similarity_matrix[:self.batch_size, self.batch_size:]  # (bs, bs)
        denominator_qk = nominator_qk + torch.sum(self.negatives_mask * torch.exp(negatives_qk/self.temperature), dim=1)
        

        nominator_kq = torch.exp(sim_kq / self.temperature)
        negatives_kq = similarity_matrix[self.batch_size:, :self.batch_size]
        denominator_kq = nominator_kq + torch.sum(self.negatives_mask * torch.exp(negatives_kq/self.temperature), dim=1)

        loss_qk = torch.sum(-torch.log(nominator_qk / denominator_qk + eps)) / self.batch_size
        loss_kq = torch.sum(-torch.log(nominator_kq / denominator_kq + eps)) / self.batch_size
        loss = loss_qk + loss_kq

        return loss
    def fcl_loss(self, z1, z2):
        batch_size=z1.shape[0]
        feature=z1.shape[1]
        bn = nn.BatchNorm1d(z1.shape[-1], affine=False).to(self.device)
        # empirical cross-correlation matrix
        c = bn(z1).T @ bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().div(feature)
        off_diag = self.off_diagonal(c).pow_(2).sum().div(feature)
        # loss = on_diag + self.single_lambd * off_diag
        loss = on_diag + 0.2 * off_diag
        return loss
    def off_diagonal(self,x):
    # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    def mcl_two_sample_loss(self, multimodal, label=True):

        batch = [i for i in range(multimodal[0].shape[0])] # 用于随机索引
        index1 = np.random.choice(batch, multimodal[0].shape[0] * (9+1), replace=True) # 随机选择索引，允许重复
        index2 = np.random.choice(batch, multimodal[0].shape[0] * (9+1), replace=True)
        index3 = np.where(index1 != index2)[0] # 找出index1 中不等于 index2 的元素的位置索引
        index1 = index1[index3]
        index2 = index2[index3] # 保留index1、index2中对应位置不相等的元素
        
        total_len = multimodal[0].shape[0] * 9
        if len(index1) > total_len:
            index1 = index1[:total_len]
            index2 = index2[:total_len]
            
        label_n = torch.zeros((len(index1),1),).float().cuda()
        label_p = torch.ones((multimodal[0].shape[0],1),).float().cuda()
        
        negative_pair1 = torch.cat([multimodal[0][index1], multimodal[1][index2]], dim=-1)
        positive_pair1 = torch.cat([multimodal[0], multimodal[1]], dim=-1)
        pair1 = torch.cat([negative_pair1, positive_pair1], dim=0)
        
        pred_y1 = self.classifier3_sample(pair1) # [1280, 1]
        pred_label = torch.cat([label_n+1, label_p-1], dim=0) # [1280, 1]
        loss1 = self.criterion(pred_y1, pred_label)
    
        return loss1
    def mcl_two_cluster_loss(self, cluster_centers_list, label=True):
        cluster_loss = 0
        num_views = len(cluster_centers_list)  # Number of views
        num_sub_lists = len(cluster_centers_list[0])  # Number of cluster centers per view
        for sub_list_index in range(num_sub_lists):
            sub_list_loss = 0
            for i in range(num_views):
                cluster_centers = cluster_centers_list[i][sub_list_index]
                num_class = [j for j in range(cluster_centers.shape[0])]
                cluster_batch_size = cluster_centers.shape[0]

                # Randomly select cluster center indices for two views
                index1 = np.random.choice(num_class, cluster_batch_size * (6 + 1), replace=True)
                index2 = np.random.choice(num_class, cluster_batch_size * (6 + 1), replace=True)

                # Ensure index1 and index2 are not equal
                valid_indices = np.where(index1 != index2)[0]
                index1 = index1[valid_indices]
                index2 = index2[valid_indices]

                # Generate third index to ensure all are different
                # index3 = np.random.choice(num_class, len(index1), replace=True)
                # valid_indices = np.intersect1d(np.where(index1 != index3)[0], np.where(index2 != index3)[0])
                # index1 = index1[valid_indices]
                # index2 = index2[valid_indices]
                # index3 = index3[valid_indices]

                total_len = cluster_batch_size * 6
                if len(index1) > total_len:
                    index1 = index1[:total_len]
                    index2 = index2[:total_len]
                    # index3 = index3[:total_len]

                # Generate negative and positive labels
                label_n = torch.zeros((len(index1), 1),).float().cuda()
                label_p = torch.ones((cluster_centers.shape[0], 1),).float().cuda()

                # Generate negative and positive pairs
                negative_pair = torch.cat([
                    cluster_centers_list[0][sub_list_index][index1],
                    cluster_centers_list[1][sub_list_index][index2]
                ], dim=-1)
                positive_pair = torch.cat([
                cluster_centers_list[0][sub_list_index],
                cluster_centers_list[1][sub_list_index]
                ], dim=-1) 

                pair = torch.cat([negative_pair, positive_pair], dim=0)

                # Calculate cluster center loss
                pred_y = self.classifier3_cluster(pair)  # [cluster_len*2, 1]
                pred_label = torch.cat([label_n + 1, label_p - 1], dim=0)  # [cluster_len*2, 1]
                sub_list_loss += self.criterion(pred_y, pred_label)  # Cluster center loss
            cluster_loss += sub_list_loss
        return cluster_loss
    def mcl_three_loss(self,multimodal,label=True):

        batch = [i for i in range(multimodal[0].shape[0])]#用于随机索引
        index1 = np.random.choice(batch, multimodal[0].shape[0] * (2+1), replace=True)#随机选择索引，允许重复
        index2 = np.random.choice(batch, multimodal[0].shape[0] * (2+1), replace=True)
        index3 = np.where(index1 != index2)[0]#找出index1 中不等于 index2 的元素的位置索引
        index1 = index1[index3]
        index2 = index2[index3]#保留index1、index2中对应位置不相等的元素
        
        index3 = np.random.choice(batch, len(index1), replace=True)
        
        index4 = np.where(index1 != index3)[0]#找到index1和index3不同的位置
        index5 = np.where(index2 != index3)[0]#找到index2和index3不同的位置
        index4 = set(index4).intersection(set(index5))#找到index1、index2、index3都不相同的位置
        index4 = list(index4)
        index1 = index1[index4]
        index2 = index2[index4]
        index3 = index3[index4]
        
        total_len = multimodal[0].shape[0] *2
        if len(index1) > total_len:
            index1 = index1[:total_len]
            index2 = index2[:total_len]
            index3 = index3[:total_len]
        label_n  = torch.zeros((len(index1),1),).float().cuda()
        label_p = torch.ones((multimodal[0].shape[0],1),).float().cuda()
        
        negative_pair1 = torch.cat([multimodal[0][index1], multimodal[1][index2], multimodal[2][index3]], dim = -1)
        positive_pair1 = torch.cat([multimodal[0], multimodal[1], multimodal[2]], dim = -1)
        pair1 = torch.cat([negative_pair1, positive_pair1], dim=0)
        
        pred_y1 = self.classifier3_sample(pair1)#[1280, 1]
        pred_label = torch.cat([label_n+1, label_p-1], dim=0)#[1280, 1]
        # loss1 = self.loss(y1, label)
        loss1 = self.criterion(pred_y1, pred_label)
      
        return loss1
    def mcl_three_cluster_loss(self, cluster_centers_list, label=True):
        cluster_loss = 0
        num_views = len(cluster_centers_list)#视图数量
        num_sub_lists = len(cluster_centers_list[0])#每个视图对应的聚类中心个数
        for sub_list_index in range(num_sub_lists):
            sub_list_loss = 0
            for i in range(num_views):
                cluster_centers = cluster_centers_list[i][sub_list_index]
                num_class = [j for j in range(cluster_centers.shape[0])]
                cluster_batch_size = cluster_centers.shape[0]
                # 随机选择聚类中心索引，生成不同视图之间的负样本和正样本
                index1 = np.random.choice(num_class, cluster_batch_size * (2 + 1), replace=True)
                index2 = np.random.choice(num_class, cluster_batch_size * (2 + 1), replace=True)
                index3 = np.where(index1!= index2)[0]  # 找出 index1 中不等于 index2 的元素
                index1 = index1[index3]
                index2 = index2[index3]

                # 再选择第三个索引，确保与前两个不同
                index3 = np.random.choice(num_class, len(index1), replace=True)
                index4 = np.where(index1!= index3)[0]  # 找到 index1 和 index3 不同的位置
                index5 = np.where(index2!= index3)[0]  # 找到 index2 和 index3 不同的位置
                index4 = set(index4).intersection(set(index5))  # 找到 index1、index2、index3 都不相同的位置
                index4 = list(index4)

                # 保证索引长度一致
                index1 = index1[index4]
                index2 = index2[index4]
                index3 = index3[index4]

                total_len = cluster_batch_size * 2
                if len(index1) > total_len:
                    index1 = index1[:total_len]
                    index2 = index2[:total_len]
                    index3 = index3[:total_len]

                # 生成负样本和正样本的标签
                label_n = torch.zeros((len(index1), 1),).float().cuda()
                label_p = torch.ones((cluster_centers.shape[0], 1),).float().cuda()

                # 生成负样本和正样本
                negative_pair = torch.cat([cluster_centers_list[0][sub_list_index][index1],
                                       cluster_centers_list[1][sub_list_index][index2],
                                       cluster_centers_list[2][sub_list_index][index3]], dim=-1)
                positive_pair = torch.cat([cluster_centers, cluster_centers, cluster_centers], dim=-1)

                pair = torch.cat([negative_pair, positive_pair], dim=0)

                # 计算聚类中心的损失
                pred_y = self.classifier3_cluster(pair) # [cluster_len*2, 1]
                pred_label = torch.cat([label_n + 1, label_p - 1], dim=0)  # [cluster_len*2, 1]
                sub_list_loss += self.criterion(pred_y, pred_label)  # 聚类中心损失
            cluster_loss += sub_list_loss
        return cluster_loss
