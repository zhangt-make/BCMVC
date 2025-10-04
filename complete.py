import os
import sys
import time
import h5py
import torch
import scipy
import random
import logging
import sklearn
import numpy as np
import torch.nn as nn
from sklearn import metrics
from scipy.io import loadmat

import torch.nn.functional as F
from sklearn.cluster import KMeans 

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.optimize import linear_sum_assignment


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pi = 3.1415926535
#######
def plot_feature_correlation_heatmap(A, save_dir, filename="zs_h(2)_Scene15_fheatmap.png"):
    """
    绘制特征相关性热力图并保存
    
    参数：
    - A: torch.Tensor, shape [batch_size, d]
    - save_dir: str, 保存目录
    - filename: str, 文件名
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError("输入 A 必须是 torch.Tensor 类型")
    
    if A.dim() != 2:
        raise ValueError("输入 A 必须是二维的，形状为 [batch_size, d]")
    
    # 将 A 转为 numpy，并构建 DataFrame
    A_np = A.detach().cpu().numpy()
    df = pd.DataFrame(A_np)
    
    # 计算特征之间的相关性矩阵（皮尔逊相关系数）
    corr_matrix = df.corr()
    
    # 画热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f",
                square=True, cbar=False)#cbar=True
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.tight_layout()
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    # 保存图片
    plt.savefig(save_path)
    plt.close()
    print(f"热力图已保存至：{save_path}")

#============================================================================================
def get_logger(file_name, data_name):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    filename = "./results/" + data_name + ".log"
    # filename = "./time/" + data_name  + ".log"
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

#=============================================================================================
# path = '../datasets/'
path = './data/'
# path = '/home/zt/work/test/test_ff/data/'
def loadData(data_name):
    if 'Caltech-5V' in data_name:
        data = scipy.io.loadmat(data_name) 
        features = np.empty((1, 5), dtype=object)
        features[0][0] = data['X1'].astype(np.float32)
        features[0][1] = data['X2'].astype(np.float32)
        features[0][2] = data['X3'].astype(np.float32)
        features[0][3] = data['X4'].astype(np.float32)
        features[0][4] = data['X5'].astype(np.float32)
        gnd = np.squeeze(data['Y']).astype(np.int32)
    elif 'Caltech-2V' in data_name:
        data_name='./data/Caltech-5V.mat'
        data = scipy.io.loadmat(data_name) 
        features = np.empty((1, 2), dtype=object)
        features[0][0] = data['X1'].astype(np.float32)
        features[0][1] = data['X2'].astype(np.float32)
        gnd = np.squeeze(data['Y']).astype(np.int32)
    elif 'Scene15' in data_name:
        data = scipy.io.loadmat(data_name) 
        features = np.empty((1, 3), dtype=object)
        features[0][0] = data['X'][0][0].astype(np.float32)
        features[0][1] = data['X'][1][0].astype(np.float32)
        features[0][2] = data['X'][2][0].astype(np.float32)
        gnd = np.squeeze(data['Y']).astype(np.int32) 
    elif 'synthetic3d' in data_name:
        data = scipy.io.loadmat(data_name) 
        features = data['X']
        features = np.transpose(features)
        gnd = data['Y']
    elif 'uci-digit' in data_name:
        data = scipy.io.loadmat(data_name) 
        features = np.empty((1, 3), dtype=object)
        features[0][0] = data['mfeat_fac']
        features[0][1] = data['mfeat_fou']
        features[0][2] = data['mfeat_kar']
        gnd = data['truth']
    elif 'ALOI100' in data_name:
        data = scipy.io.loadmat(data_name) 
        features = np.empty((1, 4), dtype=object)
        features[0][0] = data['X'][0][0].astype(np.float32)
        features[0][1] = data['X'][0][1].astype(np.float32)
        features[0][2] = data['X'][0][2].astype(np.float32)
        features[0][3] = data['X'][0][3].astype(np.float32)
        gnd = np.squeeze(data['Y']).astype(np.int32)

    elif '3sources' in data_name:
        data = scipy.io.loadmat(data_name) 
        features = np.empty((1, 3), dtype=object)
        features[0][0] = data['data'][0][0].T
        features[0][1] = data['data'][0][1].T
        features[0][2] = data['data'][0][2].T
        gnd = data['truelabel'][0][0]
    elif 'cifar100' in data_name:
        data = scipy.io.loadmat(data_name) 
        features = np.empty((1, 3), dtype=object)
        features[0][0]=data['data'][0][0].T.astype(np.float32)
        features[0][1]=data['data'][1][0].T.astype(np.float32)
        features[0][2]=data['data'][2][0].T.astype(np.float32)
        gnd=data['truelabel'][0][0].astype(np.int32).reshape(50000,)
    elif 'cifar10' in data_name:
        data = scipy.io.loadmat(data_name) 
        features = np.empty((1, 3), dtype=object)
        features[0][0] = data['data'][0][0].T.astype(np.float32)
        features[0][1] = data['data'][1][0].T.astype(np.float32)
        features[0][2] = data['data'][2][0].T.astype(np.float32)
        gnd = np.squeeze(data['truelabel'][0][0]).astype(np.int32)
    # elif 'STL10' in data_name:
    #     data=h5py.
    gnd = gnd.flatten()
    return features, gnd
class AnyDataset(Dataset):
    def __init__(self, dataname):
        self.features, self.gnd = loadData(path + dataname + '.mat')
        self.v = self.features.shape[1]
        for i in range(0, self.v):
            minmax = sklearn.preprocessing.MinMaxScaler()
            self.features[0][i] = minmax.fit_transform(self.features[0][i])

        self.iden = torch.tensor(np.identity(self.features[0][0].shape[0])).float()
        self.dataname = dataname

    def __len__(self):
        return self.gnd.shape[0]

    def __getitem__(self, idx):
        if(self.v == 2):
            return list([torch.from_numpy(np.array(self.features[0][0][idx],dtype=np.float32)),\
                   torch.from_numpy(np.array(self.features[0][1][idx],dtype=np.float32))]),torch.from_numpy(np.array(self.gnd[idx])), \
                   torch.from_numpy(np.array(idx)),torch.from_numpy(np.array(self.iden[idx]))
        if(self.v == 3):
            return list([torch.from_numpy(np.array(self.features[0][0][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][1][idx],\
                dtype=np.float32)),torch.from_numpy(np.array(self.features[0][2][idx],dtype=np.float32))]),torch.from_numpy(np.array(self.gnd[idx])), \
                    torch.from_numpy(np.array(idx)),torch.from_numpy(np.array(self.iden[idx]))
        if(self.v == 4):
            return list([torch.from_numpy(np.array(self.features[0][0][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][1][idx],\
                dtype=np.float32)),torch.from_numpy(np.array(self.features[0][2][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][3][idx],\
                    dtype=np.float32))]),torch.from_numpy(np.array(self.gnd[idx])), torch.from_numpy(np.array(idx)),torch.from_numpy(np.array(self.iden[idx]))
        if(self.v == 5):
            return list([torch.from_numpy(np.array(self.features[0][0][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][1][idx],\
                dtype=np.float32)),torch.from_numpy(np.array(self.features[0][2][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][3][idx],\
                    dtype=np.float32)),torch.from_numpy(np.array(self.features[0][4][idx],dtype=np.float32))]),torch.from_numpy(np.array(self.gnd[idx])),\
                torch.from_numpy(np.array(idx)),torch.from_numpy(np.array(self.iden[idx]))
        if(self.v == 6):
            return list([torch.from_numpy(np.array(self.features[0][0][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][1][idx],\
                dtype=np.float32)),torch.from_numpy(np.array(self.features[0][2][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][3][idx],\
                    dtype=np.float32)),torch.from_numpy(np.array(self.features[0][4][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][5][idx],\
                        dtype=np.float32))]),torch.from_numpy(np.array(self.gnd[idx])), torch.from_numpy(np.array(idx)),torch.from_numpy(np.array(self.iden[idx]))
def dataset_with_info(dataname):
    features, gnd = loadData(path + dataname + '.mat') 
    print(features.shape)
    views = max(features.shape[0],features.shape[1])
    input_num = features[0][0].shape[0]
    datasetforuse = AnyDataset(dataname)
    nc = len(np.unique(gnd))
    input_dims = []
    for v in range(views):
        dim = features[0][v].shape[1]
        input_dims.append(dim)
    print("Data: "+ dataname + ", number of data: " + str(input_num) + ", views: " + str(views) + ", clusters: " 
            + str(nc) + ", each view: ", input_dims)

    return datasetforuse, input_num, views, nc, input_dims, gnd
#====================================================================================================================
#=========================================================================================================================
    
#========================================================================================================================
class Encoder(nn.Module):
    def __init__(self, dims, bn = False):
        super(Encoder, self).__init__()
        models = []
        for i in range(len(dims) - 1):
            models.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                models.append(nn.BatchNorm1d(dims[i + 1]))
                models.append(nn.ReLU(inplace=True))
        self.models = nn.Sequential(*models)
    def forward(self, X):
        return self.models(X)
class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        models = []
        for i in range(len(dims) - 1):
            models.append(nn.Linear(dims[i], dims[i + 1]))
            if i == len(dims) - 2:
                models.append(nn.ReLU())
        self.models = nn.Sequential(*models)
    
    def forward(self, X):
        return self.models(X) 

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import  squareform
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler
class TESTNet(nn.Module):
    def __init__(self,batch_size, input_dims, view_num, low_feature_dim, number_class, h_dims=[200, 100],num_filter=2,num_layer=1,dropout=0.,mlp_ratio=4., device='cuda'):
        super().__init__()
        self.input_dims = input_dims
        self.view = view_num
        self.low_feature_dim = low_feature_dim
        self.h_dims = h_dims
        self.device = device
        self.number_class=number_class

        h_dims_reverse = list(reversed(h_dims))

        self.encoders = []
        self.decoders = []

        for v in range(self.view):
            self.encoders.append(Encoder([input_dims[v]] + h_dims + [low_feature_dim], bn=True).to(device))
            self.decoders.append(Decoder([low_feature_dim ] + h_dims_reverse + [input_dims[v]]).to(device))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
    def clustering_for_H_view(self, H, num_clusters, device):
        H_view_center_results = []
        for factor in [4, 2, 1]:
            current_num_clusters = factor * num_clusters
            H_np = H.cpu().detach().numpy()
            dist_matrix = pairwise_distances(H_np, metric='euclidean')
            dist_matrix = (dist_matrix + dist_matrix.T) / 2
            condensed_dist_matrix = squareform(dist_matrix)
            Z = linkage(condensed_dist_matrix, method='ward')
            labels = fcluster(Z, t=current_num_clusters, criterion='maxclust')

            cluster_centers = []
            for i in range(1, current_num_clusters + 1):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) > 0:
                    cluster_center = H[cluster_indices].to(device).mean(dim=0)
                    cluster_center = cluster_center.clone().detach()
                else:
                    cluster_center = torch.zeros(H.size(1)).to(device)
                    cluster_center = cluster_center.clone().detach()
                cluster_centers.append(cluster_center)
            H_view_center_results.append(torch.stack(cluster_centers))

        return H_view_center_results
    # def clustering_for_H_view(self, H, num_clusters, device):
    #     H_np = H.cpu().detach().numpy()
        
    #     # 计算距离矩阵
    #     dist_matrix = pairwise_distances(H_np, metric='euclidean')
    #     dist_matrix = (dist_matrix + dist_matrix.T) / 2  # 确保对称性
    #     condensed_dist_matrix = squareform(dist_matrix)
        
    #     # 层次聚类
    #     Z = linkage(condensed_dist_matrix, method='ward')
    #     labels = fcluster(Z, t=num_clusters, criterion='maxclust')
        
    #     # 计算聚类中心
    #     cluster_centers = []
    #     for i in range(1, num_clusters + 1):
    #         cluster_indices = np.where(labels == i)[0]
    #         if len(cluster_indices) > 0:
    #             cluster_center = H[cluster_indices].to(device).mean(dim=0)
    #             cluster_center = cluster_center.clone().detach()
    #         else:
    #             cluster_center = torch.zeros(H.size(1)).to(device)
    #             cluster_center = cluster_center.clone().detach()
    #         cluster_centers.append(cluster_center)
        
    #     return torch.stack(cluster_centers)
    def forward(self, xs,top_R,clustering=False):
        zs=[]
        zs_svd=[]
        zs_r=[]
        
        views_center=[]
        
        for v in range(self.view):
            x=xs[v]
            z=self.encoders[v](x)
            z_r=self.decoders[v](z)
            zs_r.append(z_r)
            zs.append(z)#[batch_size,d]
            z_svd=self.svd_weighting(z,device=z.device)
            view_center=self.clustering_for_H_view(z,self.number_class,self.device)
            views_center.append(view_center)
            zs_svd.append(z_svd)
        fusion_feature = torch.concat(zs_svd,dim=1)  # (batch, d_model)#[batch_size,d]
        return zs_svd, zs_r, fusion_feature,views_center
        # return zs, zs_r, fusion_feature
    def svd_weighting(self, A, device):
        """
        直接对奇异值进行加权，增强主要信号，抑制噪声。

        参数:
            A (torch.Tensor): 输入矩阵，形状为 (m, n)。
            device (torch.device): 目标设备（如 'cuda' 或 'cpu'）。

        返回:
            A_weighted (torch.Tensor): 加权后的矩阵，形状为 (m, n)。
        """
        # 将输入矩阵 A 转换为 NumPy 数组
        A_np = A.cpu().detach().numpy()

        # 1. 对矩阵 A 进行奇异值分解
        U, s, VT = np.linalg.svd(A_np, full_matrices=False)

        # 2. 计算归一化权重
        weights = s / np.sum(s)

        # 3. 对奇异值进行加权
        S_weighted = s * weights

        # 4. 重构加权矩阵
        Sigma_weighted = np.diag(S_weighted)  # 将加权后的奇异值转换为对角矩阵
        A_weighted_np = U @ Sigma_weighted @ VT

        # 将结果转换回 PyTorch 张量，并移动到目标设备
        A_weighted = torch.from_numpy(A_weighted_np).to(device)
        A_weighted.requires_grad_(True)

        return A_weighted
    # # def svd_weighting(self, A, device, top_r=None):
    # #     """
    # #     对奇异值进行加权（支持超参数 r），增强主要信号，抑制噪声。

    # #     参数:
    # #         A (torch.Tensor): 输入矩阵，形状为 (m, n)。
    # #         device (torch.device): 目标设备（如 'cuda' 或 'cpu'）。
    # #         top_r (int, optional): 保留的奇异值数量（超参数 r）。如果为 None，则保留所有奇异值。

    # #     返回:
    # #         A_weighted (torch.Tensor): 加权后的矩阵，形状为 (m, n)。
    # #     """
    # #     # 将输入矩阵 A 转换为 NumPy 数组
    # #     A_np = A.cpu().detach().numpy()

    # #     # 1. 对矩阵 A 进行奇异值分解
    # #     U, s, VT = np.linalg.svd(A_np, full_matrices=False)

    # #     # 确定 r 的值
    # #     r = min(U.shape[1], VT.shape[0])  # 原始最大 r = min(m, n)
    # #     if top_r is not None:
    # #         r = min(r, top_r)  # 截断到指定的 top_r

    # #     # 截断 U, s, VT
    # #     U_trunc = U[:, :r]
    # #     s_trunc = s[:r]
    # #     VT_trunc = VT[:r, :]

    # #     # 2. 计算归一化权重
    # #     weights = s_trunc / np.sum(s_trunc)

    # #     # 3. 对奇异值进行加权
    # #     S_weighted = s_trunc * weights

    # #     # 4. 重构加权矩阵
    # #     Sigma_weighted = np.diag(S_weighted)  # 将加权后的奇异值转换为对角矩阵
    # #     A_weighted_np = U_trunc @ Sigma_weighted @ VT_trunc

    # #     # 将结果转换回 PyTorch 张量，并移动到目标设备
    # #     A_weighted = torch.from_numpy(A_weighted_np).to(device)
    # #     A_weighted.requires_grad_(True)

    # #     return A_weighted
    # import torch

    # # def svd_weighting(self, A, device, top_r=None):
    # #     """
    # #     对奇异值进行加权（支持超参数 r），增强主要信号，抑制噪声，支持反向传播。

    # #     参数:
    # #         A (torch.Tensor): 输入矩阵，形状为 (m, n)。
    # #         device (torch.device): 目标设备（如 'cuda' 或 'cpu'）。
    # #         top_r (int, optional): 保留的奇异值数量（超参数 r）。如果为 None，则保留所有奇异值。

    # #     返回:
    # #         A_weighted (torch.Tensor): 加权后的矩阵，形状为 (m, n)。
    # #     """
    # #     # 1. PyTorch SVD
    # #     U, S, Vh = torch.linalg.svd(A, full_matrices=False)  # S形状是 (min(m,n),)

    # #     # 2. 确定 r 的值
    # #     r = S.size(0)
    # #     if top_r is not None:
    # #         r = min(r, top_r)

    # #     # 3. 截断
    # #     U_trunc = U[:, :r]            # (m, r)
    # #     S_trunc = S[:r]               # (r,)
    # #     Vh_trunc = Vh[:r, :]          # (r, n)

    # #     # 4. 计算归一化权重
    # #     weights = S_trunc / torch.sum(S_trunc)

    # #     # 5. 加权奇异值
    # #     S_weighted = S_trunc * weights

    # #     # 6. 重构矩阵
    # #     Sigma_weighted = torch.diag(S_weighted)  # (r, r)
    # #     A_weighted = U_trunc @ Sigma_weighted @ Vh_trunc  # (m, n)

    # #     # 7. 确保返回的张量在正确设备
    # #     A_weighted = A_weighted.to(device)

    # #     return A_weighted



    # # def svd_weighting(self, A, device, energy_threshold=0.95):
    # #     # 将输入张量转换为NumPy数组
    # #     A = A.cpu().detach().numpy()    
    # #     U, s, VT = np.linalg.svd(A)
        
    # #     # 计算奇异值的累积能量贡献
    # #     total_energy = np.sum(s**2)  # 总能量
    # #     cumulative_energy = np.cumsum(s**2)  # 累积能量
    # #     normalized_cumulative_energy = cumulative_energy / total_energy  # 归一化累积能量
        
    # #     # 找到满足能量贡献阈值的最小奇异值数量
    # #     num_components = np.argmax(normalized_cumulative_energy >= energy_threshold) + 1
        
    # #     # 对奇异值进行加权处理
    # #     s_weighted = s.copy()  # 复制奇异值以避免修改原数组
    # #     s_weighted[:num_components] *= 1.5  # 大于阈值的奇异值乘以1.5
    # #     s_weighted[num_components:] *= 0.5  # 小于阈值的奇异值乘以0.5
        
    # #     # 计算权重（基于加权后的奇异值）
    # #     weights = s_weighted / s_weighted.sum()
        
    # #     # 确保weights的形状与U、VT的列数匹配
    # #     num_components = min(U.shape[1], VT.shape[0])
    # #     weights = weights[:num_components]
    # #     weights = weights / weights.sum()  # 再次归一化
        
    # #     # 对U和VT进行加权
    # #     U_weighted = U[:, :num_components] * weights.reshape(1, -1)
    # #     VT_weighted = (VT[:num_components, :].T * weights.reshape(1, -1)).T
        
    # #     # 构建加权后的Sigma矩阵
    # #     Sigma_weighted = np.diag(s_weighted[:num_components])
        
    # #     # 重建加权后的矩阵
    # #     A_weighted = U_weighted @ Sigma_weighted @ VT_weighted
        
    # #     # 将结果转换回PyTorch张量并移动到指定设备
    # #     A_weighted = torch.from_numpy(A_weighted)
    # #     A_weighted = A_weighted.to(device)
    # #     A_weighted.requires_grad_(True)
        
    # #     return A_weighted
    # def svd_weighting_reverse(self, A, device):
    #     A = A.cpu().detach().numpy()
    #     U, s, VT = np.linalg.svd(A)
    #     Sigma = np.diag(s)

    #     # 计算新的反向权重
    #     weights = s[::-1] / s.sum()

    #     num_components = min(U.shape[1], VT.shape[0])
    #     weights = weights[:num_components]
    #     weights = weights / weights.sum()
    #     U_weighted = U[:, :num_components] * weights.reshape(1, -1)
    #     VT_weighted = (VT[:num_components, :].T * weights.reshape(1, -1)).T

    #     A_weighted = U_weighted @ Sigma[:num_components, :num_components] @ VT_weighted
    #     A_weighted = torch.from_numpy(A_weighted)
    #     A_weighted = A_weighted.to(device)
    #     A_weighted.requires_grad_(True)
    #     return A_weighted
    # def svd_weighting_two(self, A, device):
    #     A = A.cpu().detach().numpy()
    #     U, s, VT = np.linalg.svd(A)

    #     num_components = min(U.shape[1], VT.shape[0])
    #     s = s[:num_components]
    #     U = U[:, :num_components]
    #     VT = VT[:num_components, :]

    #     # 计算权重
    #     weights = s / s.sum()
    #     # 创建加权对角矩阵
    #     Sigma_weighted = np.diag(weights * s)
    #     # 重构加权后的矩阵
    #     A_weighted = U @ Sigma_weighted @ VT

    #     A_weighted = torch.from_numpy(A_weighted)
    #     A_weighted = A_weighted.to(device)
    #     A_weighted.requires_grad_(True)

    #     return A_weighted
    # def svd_weighting_test(self, A, device):
    #     # 确保 A 是一个 PyTorch 张量并移动到所需设备
    #     A = A.to(device)

    #     # 进行奇异值分解
    #     U, s, VT = torch.svd(A)

    #     # 计算求和权重
    #     weights = s / torch.sum(s)

    #     # 对奇异值进行求和加权
    #     adjusted_s = s * weights

    #     # 构造新的对角矩阵
    #     Sigma_adjusted = torch.diag(adjusted_s)

    #     # 正确的矩阵乘法，使用 Sigma_adjusted 作为对角矩阵
    #     A_weighted = torch.mm(U, torch.mm(Sigma_adjusted, VT.t()))
    #     return A_weighted

#==========================================================================================================================
seed = 10#10

def set_seed(seed=10):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    
dataname = "uci-digit"#LandUse-21(2100),WikipediaArticles(693),BBCSport(544),synthetic3d(600),prokaryotic(551),handwritten(2000),100leaves(1600),handwritten、Mfeat(2000)
#Caltech101-7(1474)、MSRCV1(210)、3sources(169)、Hdigit(10000)、ORL(400)、NGs(500)、webkb(203)、Cora(2708)、ALOI100(10800)、BBC4view_685(685)、Reuters_21578(1500)
logger = get_logger(__file__, dataname)
datasetforuse, data_size, view_num, nc, input_dims, gnd = dataset_with_info(dataname)

##写入数据
# epochs=[]
# acc_values=[]
# nmi_values=[]
# loss_values=[]



def main():  
    
    temperature=1
    batch_size =400
    
    low_feature_dim = 256#256  
                     
    h_dims = [1024, 1024]      #h_dims = [1024, 1024]        
    device="cuda:0"#device="cuda:0"
    train_loader = DataLoader(datasetforuse, batch_size=batch_size, shuffle=True, drop_last=False) #drop_last=False
    test_loader = DataLoader(datasetforuse, batch_size=batch_size, shuffle=False, drop_last=False)#drop_last=False
    model = TESTNet(batch_size,input_dims, view_num,low_feature_dim, nc, h_dims=h_dims,num_filter=2,num_layer=1,dropout=0.,mlp_ratio=4.,device=device )
    model.to(device)
    
    
    
    
    lr = 0.0002#0.0001
    mcl_weight=1
    mcl_cluster_weight=10
    
    # bcl_weight=1
    
    
    # parameters =model.parameters()
    # optimizer = torch.optim.Adam(parameters, lr=lr)
    
    selfcontrastiveLoss=SELFContrastiveLoss(batch_size,temperature,low_feature_dim,view_num,device).to(device)
    
    
    
    #优化
    model_params = list(model.parameters())
    contrastive_params = list(selfcontrastiveLoss.classifier3_sample.parameters())+list(selfcontrastiveLoss.classifier3_cluster.parameters())
    # 为不同的参数组设置不同的学习率
    optimizer = torch.optim.Adam([
        {'params': model_params, 'lr': lr},
        {'params': contrastive_params, 'lr': 0.000001}
    ])
    
    
    train_epoch =120#120
    top_R=40
    
    settings = {"seed": seed, "batch_size": batch_size, "dataname": dataname, "learning_rate": lr, "h_dims": h_dims, "low_feature_dim": low_feature_dim, 
                "epochs": train_epoch,"mcl_weight":mcl_weight,"mcl_cluster_weight":mcl_cluster_weight,"k":4,"device": device}
    logger.info(str(settings))
    
    losses = []
    ########
    epoch_metrics={
        'ACCo':[],
        'NMIo':[],
        'Purityo':[],
        'ARIo':[],
        'Fscoreo':[],
                }
    ############
    
    for epoch in range(train_epoch):
        total_loss = 0.
        rec_loss = torch.nn.MSELoss()
        for xs, _, idx, inpu in train_loader:
        
            
            model.train()
            for v in range(view_num):
                xs[v] = xs[v].to(device)
            zs_outputs, zs_r, fusion_feature,views_center = model(xs,top_R, clustering=False)#[batch_size,d//2+1],[batch_size,d//2+1],[batch_size,view*(d//2+1)],
            # criterion_self = SelfContrastiveLoss(batch_size=batch_size)

            # plot_feature_correlation_heatmap(zs_outputs[2],save_dir="/home/zt/work/low_rank/heatmap")
            optimizer.zero_grad()
            loss_list=[]
            
            if view_num==2:
                loss_list.append(mcl_weight*selfcontrastiveLoss.mcl_two_sample_loss(zs_outputs))#0.8604
                loss_list.append(mcl_cluster_weight*selfcontrastiveLoss.mcl_two_cluster_loss(views_center))
            elif view_num==3:
                loss_list.append(mcl_weight*selfcontrastiveLoss.mcl_three_loss(zs_outputs))
                loss_list.append(mcl_cluster_weight*selfcontrastiveLoss.mcl_three_cluster_loss(views_center))
            for v in range(view_num):
                loss_list.append(rec_loss(xs[v],zs_r[v]))
                
               
            loss = sum(loss_list)
            
            total_loss += loss.item() 
            loss.backward()
            optimizer.step()
        losses.append(total_loss)

        # print(f'epoch: {epoch+1}, total loss: {loss.item():.8f}, rec loss: {loss_rec:.8f}, min loss: {loss_min:.8f}')
        print(f'epoch: {epoch+1}, total loss: {loss.item():.8f}')
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                learned_features = []
                for x, _, idx, inpu in test_loader:
                    for v in range(view_num):
                        x[v] = x[v].to(device)
                    model.eval()
                    zs_outputs,zs_r,  fusion_feature,views_center= model(x,top_R)

                    learned_feature = fusion_feature
                    learned_features.extend(learned_feature.detach().cpu().numpy())
                    
                
                learned_features = np.array(learned_features)
                kmeans = KMeans(n_clusters=nc, n_init=50)
                y_pred = kmeans.fit_predict(learned_features)      
                ACC, NMI, Purity, ARI, Fscore = clusteringMetrics(gnd, y_pred) 

                # ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetrics(gnd, final_labels)  
                # info = {"epoch": epoch + 1, "acc": '%.4f'%ACC, "nmi": '%.4f'%NMI, "ari": '%.4f'%ARI, "Purity": '%.4f'%Purity, "fscore": '%.4f'%Fscore}
                # logger.info(str(info))
                
                
                #######.csv文件
                # epochs.append(epoch)
                # acc_values.append(ACC)
                # nmi_values.append(NMI)
                # loss_values.append(total_loss)
                
                
                
                print('epoch{}'.format(epoch+1),'ACC:{:.5f}'.format(ACC),'NMI:{:.5f}'.format(NMI),'Purity:{:.5f}'.format(Purity),\
                    'ARI:{:.5f}'.format(ARI),'Fscore:{:.5f}'.format(Fscore))
                
                epoch_metrics['ACCo'].append(ACC)
                epoch_metrics['NMIo'].append(NMI)
                epoch_metrics['Purityo'].append(Purity)
                epoch_metrics['ARIo'].append(ARI)
                epoch_metrics['Fscoreo'].append(Fscore)
    max_value={}
    for metric,values in epoch_metrics.items():
        max_value[metric]=max(values)
    max_value_str = ', '.join([f"{metric}:{value:.5f}" for metric, value in max_value.items()])
    print(max_value_str)
    logger.info(max_value_str)
    # for metric,value in max_value.items():
    #     logger.info(f"{metric}:{value}")

                # if (epoch + 1) == train_epoch:
                #     with open(f'./losses/losses_{dataname}.txt', 'a') as f:
                #         f.write(str(losses) + '\n')
                #     return ACC, NMI, Purity, ARI, Fscore, Precision, Recall
    
    ####创建DataFrame
    # data={
    #     'Epoch':epochs,
    #     'ACC':acc_values,
    #     'NMI':nmi_values,
    #     'Loss':loss_values
    #     }
    # df=pd.DataFrame(data)
    # save_directory = '/home/zt/work/low_rank/train_data'
    # csv_file_name = f'training_results_{dataname}.csv'
    # csv_file_path = os.path.join(save_directory, csv_file_name)
    # df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':
    acc_list = []
    nmi_list = []
    ari_list = []
    pur_list = []
    fscore_list  = []

    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Main function execution time: {execution_time:.4f} seconds")
    info={f"Main function execution time_ours: {execution_time:.4f} seconds"}
    logger.info(str(info))

