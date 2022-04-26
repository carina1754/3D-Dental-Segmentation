import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, n_pts, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    n_pts = x.size(2)
    x = x.view(batch_size, -1, n_pts)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, n_pts, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*n_pts

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, n_pts, num_dims)  -> (batch_size*n_pts, num_dims) #   batch_size * n_pts * k + range(0, batch_size*n_pts)
    feature = x.view(batch_size*n_pts, -1)[idx, :]
    feature = feature.view(batch_size, n_pts, k, num_dims) 
    x = x.view(batch_size, n_pts, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, n_pts, k)

class DGCNN_Seg(nn.Module):
    def __init__(self, num_classes = 24, num_neighbor = 20):
        super(DGCNN_Seg, self).__init__()

        self.k = num_neighbor
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Conv1d(256, num_classes, kernel_size=1, bias=False)
        

    def forward(self, x):
        batch_size = x.size(0)
        n_pts = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)   # (batch_size, 9, n_pts) -> (batch_size, 9*2, n_pts, k)
        x = self.conv1(x)                       # (batch_size, 9*2, n_pts, k) -> (batch_size, 64, n_pts, k)
        x = self.conv2(x)                       # (batch_size, 64, n_pts, k) -> (batch_size, 64, n_pts, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, n_pts, k) -> (batch_size, 64, n_pts)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, n_pts) -> (batch_size, 64*2, n_pts, k)
        x = self.conv3(x)                       # (batch_size, 64*2, n_pts, k) -> (batch_size, 64, n_pts, k)
        x = self.conv4(x)                       # (batch_size, 64, n_pts, k) -> (batch_size, 64, n_pts, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, n_pts, k) -> (batch_size, 64, n_pts)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, n_pts) -> (batch_size, 64*2, n_pts, k)
        x = self.conv5(x)                       # (batch_size, 64*2, n_pts, k) -> (batch_size, 64, n_pts, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, n_pts, k) -> (batch_size, 64, n_pts)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, n_pts)

        x = self.conv6(x)                       # (batch_size, 64*3, n_pts) -> (batch_size, emb_dims, n_pts)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, n_pts) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, n_pts)          # (batch_size, 1024, n_pts)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, n_pts)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, n_pts) -> (batch_size, 512, n_pts)
        x = self.conv8(x)                       # (batch_size, 512, n_pts) -> (batch_size, 256, n_pts)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, n_pts) -> (batch_size, 13, n_pts)
        
        return x
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dgcnn = DGCNN_Seg().to(device)
    summary(dgcnn, (24, 5000))
