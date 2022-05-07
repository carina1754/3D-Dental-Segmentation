import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torchsummary import summary as summary_
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.get_device())
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.get_device())
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class MeshSegNet_U(nn.Module):
    def __init__(self, num_classes=15, num_channels=15, with_dropout=True, dropout_p=0.5):
        super(MeshSegNet_U, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.with_dropout = with_dropout
        self.dropout_p = dropout_p

        # MLP-1 [64, 64]
        self.mlp1_conv1 = torch.nn.Conv1d(self.num_channels, 64, 1)
        self.mlp1_conv2 = torch.nn.Conv1d(64, 64, 1)
        self.mlp1_bn1 = nn.BatchNorm1d(64)
        self.mlp1_bn2 = nn.BatchNorm1d(64)
        # FTM (feature-transformer module)
        self.fstn = STNkd(k=64)
        # GLM-1 (graph-contrained learning modulus)
        self.glm1_conv1_1 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_conv1_2 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_bn1_1 = nn.BatchNorm1d(32)
        self.glm1_bn1_2 = nn.BatchNorm1d(32)
        self.glm1_conv2 = torch.nn.Conv1d(32+32, 64, 1)
        self.glm1_bn2 = nn.BatchNorm1d(64)
        # MLP-2
        self.mlp2_conv1 = torch.nn.Conv1d(64, 64, 1)
        self.mlp2_bn1 = nn.BatchNorm1d(64)
        self.mlp2_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.mlp2_bn2 = nn.BatchNorm1d(128)
        self.mlp2_conv3 = torch.nn.Conv1d(128, 512, 1)
        self.mlp2_bn3 = nn.BatchNorm1d(512)
        # GLM-2 (graph-contrained learning modulus)
        self.glm2_conv1_1 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_2 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_3 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_bn1_1 = nn.BatchNorm1d(128)
        self.glm2_bn1_2 = nn.BatchNorm1d(128)
        self.glm2_bn1_3 = nn.BatchNorm1d(128)
        self.glm2_conv2 = torch.nn.Conv1d(128*3, 512, 1)
        self.glm2_bn2 = nn.BatchNorm1d(512)
        # GLM-3 (graph-contrained learning modulus)
        self.glm3_conv1_1 = torch.nn.Conv1d(512, 128, 1)
        self.glm3_conv1_2 = torch.nn.Conv1d(512, 128, 1)
        self.glm3_conv1_3 = torch.nn.Conv1d(512, 128, 1)
        self.glm3_conv1_4 = torch.nn.Conv1d(512, 128, 1)
        self.glm3_bn1_1 = nn.BatchNorm1d(128)
        self.glm3_bn1_2 = nn.BatchNorm1d(128)
        self.glm3_bn1_3 = nn.BatchNorm1d(128)
        self.glm3_bn1_4 = nn.BatchNorm1d(128)
        self.glm3_conv2 = torch.nn.Conv1d(128*4, 512, 1)
        self.glm3_bn2 = nn.BatchNorm1d(512)
        # MLP-4
        self.mlp4_conv1 = torch.nn.Conv1d(64+512+512+512, 256, 1)
        self.mlp4_conv2 = torch.nn.Conv1d(256, 256, 1)
        self.mlp4_bn1_1 = nn.BatchNorm1d(256)
        self.mlp4_bn1_2 = nn.BatchNorm1d(256)
        self.mlp4_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.mlp4_conv4 = torch.nn.Conv1d(128, 128, 1)
        self.mlp4_bn2_1 = nn.BatchNorm1d(128)
        self.mlp4_bn2_2 = nn.BatchNorm1d(128)
        # output
        self.output_conv = torch.nn.Conv1d(128, self.num_classes, 1)
        if self.with_dropout:
            self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x, a_s, a_l,a_el):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # MLP-1
        x = F.relu(self.mlp1_bn1(self.mlp1_conv1(x)))
        x = F.relu(self.mlp1_bn2(self.mlp1_conv2(x)))
        # FTM
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x_ftm = torch.bmm(x, trans_feat)
        # GLM-1
        sap = torch.bmm(a_s, x_ftm)
        sap = sap.transpose(2, 1)
        x_ftm = x_ftm.transpose(2, 1)
        x = F.relu(self.glm1_bn1_1(self.glm1_conv1_1(x_ftm)))
        glm_1_sap = F.relu(self.glm1_bn1_2(self.glm1_conv1_2(sap)))
        x = torch.cat([x, glm_1_sap], dim=1)
        x = F.relu(self.glm1_bn2(self.glm1_conv2(x)))
        # MLP-2
        x = F.relu(self.mlp2_bn1(self.mlp2_conv1(x)))
        x = F.relu(self.mlp2_bn2(self.mlp2_conv2(x)))
        x_mlp2 = F.relu(self.mlp2_bn3(self.mlp2_conv3(x)))
        if self.with_dropout:
            x_mlp2 = self.dropout(x_mlp2)
        # GLM-2
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = torch.bmm(a_s, x_mlp2)
        sap_2 = torch.bmm(a_l, x_mlp2)
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = sap_1.transpose(2, 1)
        sap_2 = sap_2.transpose(2, 1)
        x = F.relu(self.glm2_bn1_1(self.glm2_conv1_1(x_mlp2)))
        glm_2_sap_1 = F.relu(self.glm2_bn1_2(self.glm2_conv1_2(sap_1)))
        glm_2_sap_2 = F.relu(self.glm2_bn1_3(self.glm2_conv1_3(sap_2)))
        x = torch.cat([x, glm_2_sap_1, glm_2_sap_2], dim=1)
        x = F.relu(self.glm2_bn2(self.glm2_conv2(x)))

        # GLM-3
        x = x.transpose(2, 1)
        sap_1 = torch.bmm(a_s, x)
        sap_2 = torch.bmm(a_l, x)
        sap_3 = torch.bmm(a_el, x)
        x = x.transpose(2, 1)
        sap_1 = sap_1.transpose(2, 1)
        sap_2 = sap_2.transpose(2, 1)
        sap_3 = sap_3.transpose(2, 1)
        x = F.relu(self.glm3_bn1_1(self.glm3_conv1_1(x)))
        glm_3_sap_1 = F.relu(self.glm3_bn1_2(self.glm3_conv1_2(sap_1)))
        glm_3_sap_2 = F.relu(self.glm3_bn1_3(self.glm3_conv1_3(sap_2)))
        glm_3_sap_3 = F.relu(self.glm3_bn1_4(self.glm3_conv1_4(sap_3)))
        x = torch.cat([x, glm_3_sap_1, glm_3_sap_2, glm_3_sap_3], dim=1)
        x_glm3 = F.relu(self.glm3_bn2(self.glm3_conv2(x)))
        # GMP
        x = torch.max(x_glm3, 2, keepdim=True)[0]
        # Upsample
        x = torch.nn.Upsample(n_pts)(x)
        # Dense fusion
        x = torch.cat([x, x_ftm, x_mlp2, x_glm3], dim=1)
        # MLP-4
        x = F.relu(self.mlp4_bn1_1(self.mlp4_conv1(x)))
        x = F.relu(self.mlp4_bn1_2(self.mlp4_conv2(x)))
        x = F.relu(self.mlp4_bn2_1(self.mlp4_conv3(x)))
        if self.with_dropout:
            x = self.dropout(x)
        x = F.relu(self.mlp4_bn2_2(self.mlp4_conv4(x)))
        # output
        x = self.output_conv(x)
        x = x.transpose(2,1).contiguous()
        x = torch.nn.Softmax(dim=-1)(x.view(-1, self.num_classes))
        x = x.view(batchsize, n_pts, self.num_classes)

        return x

class MeshSegNet_L(nn.Module):
    def __init__(self, num_classes=15, num_channels=15, with_dropout=True, dropout_p=0.5):
        super(MeshSegNet_L, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.with_dropout = with_dropout
        self.dropout_p = dropout_p

        # MLP-1 [64, 64]
        self.mlp1_conv1 = torch.nn.Conv1d(self.num_channels, 64, 1)
        self.mlp1_conv2 = torch.nn.Conv1d(64, 64, 1)
        self.mlp1_bn1 = nn.BatchNorm1d(64)
        self.mlp1_bn2 = nn.BatchNorm1d(64)
        # FTM (feature-transformer module)
        self.fstn = STNkd(k=64)
        # GLM-1 (graph-contrained learning modulus)
        self.glm1_conv1_1 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_conv1_2 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_bn1_1 = nn.BatchNorm1d(32)
        self.glm1_bn1_2 = nn.BatchNorm1d(32)
        self.glm1_conv2 = torch.nn.Conv1d(32+32, 64, 1)
        self.glm1_bn2 = nn.BatchNorm1d(64)
        # MLP-2
        self.mlp2_conv1 = torch.nn.Conv1d(64, 64, 1)
        self.mlp2_bn1 = nn.BatchNorm1d(64)
        self.mlp2_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.mlp2_bn2 = nn.BatchNorm1d(128)
        self.mlp2_conv3 = torch.nn.Conv1d(128, 512, 1)
        self.mlp2_bn3 = nn.BatchNorm1d(512)
        # GLM-2 (graph-contrained learning modulus)
        self.glm2_conv1_1 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_2 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_3 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_bn1_1 = nn.BatchNorm1d(128)
        self.glm2_bn1_2 = nn.BatchNorm1d(128)
        self.glm2_bn1_3 = nn.BatchNorm1d(128)
        self.glm2_conv2 = torch.nn.Conv1d(128*3, 512, 1)
        self.glm2_bn2 = nn.BatchNorm1d(512)
        # MLP-3
        self.mlp3_conv1 = torch.nn.Conv1d(64+512+512+512, 256, 1)
        self.mlp3_conv2 = torch.nn.Conv1d(256, 256, 1)
        self.mlp3_bn1_1 = nn.BatchNorm1d(256)
        self.mlp3_bn1_2 = nn.BatchNorm1d(256)
        self.mlp3_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.mlp3_conv4 = torch.nn.Conv1d(128, 128, 1)
        self.mlp3_bn2_1 = nn.BatchNorm1d(128)
        self.mlp3_bn2_2 = nn.BatchNorm1d(128)
        # output
        self.output_conv = torch.nn.Conv1d(128, self.num_classes, 1)
        if self.with_dropout:
            self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x, a_s, a_l):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # MLP-1
        x = F.relu(self.mlp1_bn1(self.mlp1_conv1(x)))
        x = F.relu(self.mlp1_bn2(self.mlp1_conv2(x)))
        # FTM
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x_ftm = torch.bmm(x, trans_feat)
        # GLM-1
        sap = torch.bmm(a_s, x_ftm)
        sap = sap.transpose(2, 1)
        x_ftm = x_ftm.transpose(2, 1)
        x = F.relu(self.glm1_bn1_1(self.glm1_conv1_1(x_ftm)))
        glm_1_sap = F.relu(self.glm1_bn1_2(self.glm1_conv1_2(sap)))
        x = torch.cat([x, glm_1_sap], dim=1)
        x = F.relu(self.glm1_bn2(self.glm1_conv2(x)))
        # MLP-2
        x = F.relu(self.mlp2_bn1(self.mlp2_conv1(x)))
        x = F.relu(self.mlp2_bn2(self.mlp2_conv2(x)))
        x_mlp2 = F.relu(self.mlp2_bn3(self.mlp2_conv3(x)))
        if self.with_dropout:
            x_mlp2 = self.dropout(x_mlp2)
        # GLM-2
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = torch.bmm(a_s, x_mlp2)
        sap_2 = torch.bmm(a_l, x_mlp2)
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = sap_1.transpose(2, 1)
        sap_2 = sap_2.transpose(2, 1)
        x = F.relu(self.glm2_bn1_1(self.glm2_conv1_1(x_mlp2)))
        glm_2_sap_1 = F.relu(self.glm2_bn1_2(self.glm2_conv1_2(sap_1)))
        glm_2_sap_2 = F.relu(self.glm2_bn1_3(self.glm2_conv1_3(sap_2)))
        x = torch.cat([x, glm_2_sap_1, glm_2_sap_2], dim=1)
        x_glm2 = F.relu(self.glm2_bn2(self.glm2_conv2(x)))
        # GMP
        x = torch.max(x_glm2, 2, keepdim=True)[0]
        # Upsample
        x = torch.nn.Upsample(n_pts)(x)
        # Dense fusion
        x = torch.cat([x, x_ftm, x_mlp2, x_glm2], dim=1)
        # MLP-3
        x = F.relu(self.mlp3_bn1_1(self.mlp3_conv1(x)))
        x = F.relu(self.mlp3_bn1_2(self.mlp3_conv2(x)))
        x = F.relu(self.mlp3_bn2_1(self.mlp3_conv3(x)))
        if self.with_dropout:
            x = self.dropout(x)
        x = F.relu(self.mlp3_bn2_2(self.mlp3_conv4(x)))
        # output
        x = self.output_conv(x)
        x = x.transpose(2,1).contiguous()
        x = torch.nn.Softmax(dim=-1)(x.view(-1, self.num_classes))
        x = x.view(batchsize, n_pts, self.num_classes)

        return x

class MeshSegNet_G(nn.Module):
    def __init__(self, num_classes=15, num_channels=15, with_dropout=True, dropout_p=0.5):
        super(MeshSegNet_G, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.with_dropout = with_dropout
        self.dropout_p = dropout_p
        self.k = 20
        # MLP-1 [64, 64]
        self.mlp1_conv1 = torch.nn.Conv1d(self.num_channels, 64, 1)
        self.mlp1_conv2 = torch.nn.Conv1d(64, 64, 1)
        self.mlp1_bn1 = nn.BatchNorm1d(64)
        self.mlp1_bn2 = nn.BatchNorm1d(64)
        # FTM (feature-transformer module)
        self.fstn = STNkd(k=64)
        # GLM-1 (graph-contrained learning modulus)
        self.glm1_conv1_1 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_conv1_2 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_bn1_1 = nn.BatchNorm1d(32)
        self.glm1_bn1_2 = nn.BatchNorm1d(32)
        self.glm1_conv2 = torch.nn.Conv1d(32+32, 64, 1)
        self.glm1_bn2 = nn.BatchNorm1d(64)
        # MLP-2
        self.mlp2_conv1 = torch.nn.Conv1d(64, 64, 1)
        self.mlp2_bn1 = nn.BatchNorm1d(64)
        self.mlp2_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.mlp2_bn2 = nn.BatchNorm1d(128)
        self.mlp2_conv3 = torch.nn.Conv1d(128, 512, 1)
        self.mlp2_bn3 = nn.BatchNorm1d(512)
        # GLM-2 (graph-contrained learning modulus)
        self.glm2_conv1_1 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_2 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_3 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_bn1_1 = nn.BatchNorm1d(128)
        self.glm2_bn1_2 = nn.BatchNorm1d(128)
        self.glm2_bn1_3 = nn.BatchNorm1d(128)
        self.glm2_conv2 = torch.nn.Conv1d(128*3, 512, 1)
        self.glm2_bn2 = nn.BatchNorm1d(512)
        # GLM-3 (graph-contrained learning modulus)
        self.glm3_conv1_1 = torch.nn.Conv1d(512, 128, 1)
        self.glm3_conv1_2 = torch.nn.Conv1d(512, 128, 1)
        self.glm3_conv1_3 = torch.nn.Conv1d(512, 128, 1)
        self.glm3_conv1_4 = torch.nn.Conv1d(512, 128, 1)
        self.glm3_bn1_1 = nn.BatchNorm1d(128)
        self.glm3_bn1_2 = nn.BatchNorm1d(128)
        self.glm3_bn1_3 = nn.BatchNorm1d(128)
        self.glm3_bn1_4 = nn.BatchNorm1d(128)
        self.glm3_conv2 = torch.nn.Conv1d(128*4, 512, 1)
        self.glm3_bn2 = nn.BatchNorm1d(512)
        # MLP-4
        self.mlp4_conv1 = torch.nn.Conv1d(64+512+512+512+512, 256, 1)
        self.mlp4_conv2 = torch.nn.Conv1d(256, 256, 1)
        self.mlp4_bn1_1 = nn.BatchNorm1d(256)
        self.mlp4_bn1_2 = nn.BatchNorm1d(256)
        self.mlp4_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.mlp4_conv4 = torch.nn.Conv1d(128, 128, 1)
        self.mlp4_bn2_1 = nn.BatchNorm1d(128)
        self.mlp4_bn2_2 = nn.BatchNorm1d(128)
        #DGCNN
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(512)
	
        self.conv1 = nn.Sequential(nn.Conv2d(30, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(512*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        # output
        self.output_conv = torch.nn.Conv1d(128, self.num_classes, 1)
        if self.with_dropout:
            self.dropout = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x, a_s, a_l):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        g = x.clone().detach() #b
        # MLP-1
        x = F.relu(self.mlp1_bn1(self.mlp1_conv1(x)))
        x = F.relu(self.mlp1_bn2(self.mlp1_conv2(x)))
        # FTM
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x_ftm = torch.bmm(x, trans_feat)
        # GLM-1
        # print(a_s1.shape,x_ftm.shape)
        sap = torch.bmm(a_s, x_ftm)
        sap = sap.transpose(2, 1)
        x_ftm = x_ftm.transpose(2, 1)
        x = F.relu(self.glm1_bn1_1(self.glm1_conv1_1(x_ftm)))
        glm_1_sap = F.relu(self.glm1_bn1_2(self.glm1_conv1_2(sap)))
        x = torch.cat([x, glm_1_sap], dim=1)
        x = F.relu(self.glm1_bn2(self.glm1_conv2(x)))
        # MLP-2
        x = F.relu(self.mlp2_bn1(self.mlp2_conv1(x)))
        x = F.relu(self.mlp2_bn2(self.mlp2_conv2(x)))
        x_mlp2 = F.relu(self.mlp2_bn3(self.mlp2_conv3(x)))
        if self.with_dropout:
            x_mlp2 = self.dropout(x_mlp2)
        # GLM-2
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = torch.bmm(a_s, x_mlp2)
        sap_2 = torch.bmm(a_l, x_mlp2)
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = sap_1.transpose(2, 1)
        sap_2 = sap_2.transpose(2, 1)
        x = F.relu(self.glm2_bn1_1(self.glm2_conv1_1(x_mlp2)))
        glm_2_sap_1 = F.relu(self.glm2_bn1_2(self.glm2_conv1_2(sap_1)))
        glm_2_sap_2 = F.relu(self.glm2_bn1_3(self.glm2_conv1_3(sap_2)))
        x = torch.cat([x, glm_2_sap_1, glm_2_sap_2], dim=1)
        x_glm2 = F.relu(self.glm2_bn2(self.glm2_conv2(x)))
        # GMP
        x = torch.max(x_glm2, 2, keepdim=True)[0]
        # Upsample
        x = torch.nn.Upsample(n_pts)(x)
        # DGCNN
        g = get_graph_feature(g, k=self.k)
        g = self.conv1(g)
        g1 = g.max(dim=-1, keepdim=False)[0]
        g = get_graph_feature(g1, k=self.k)
        g = self.conv2(g)
        g2 = g.max(dim=-1, keepdim=False)[0]
        g = get_graph_feature(g2, k=self.k)
        g = self.conv3(g)
        g3 = g.max(dim=-1, keepdim=False)[0]
        g = get_graph_feature(g3, k=self.k)
        g = self.conv4(g)
        g4 = g.max(dim=-1, keepdim=False)[0]
        g = torch.cat((g1, g2, g3, g4), dim=1)
        g = self.conv5(g)
        # Dense fusion
        x = torch.cat([x, x_ftm, x_mlp2, x_glm2,g], dim=1)
        # MLP-4
        x = F.relu(self.mlp4_bn1_1(self.mlp4_conv1(x)))
        x = F.relu(self.mlp4_bn1_2(self.mlp4_conv2(x)))
        x = F.relu(self.mlp4_bn2_1(self.mlp4_conv3(x)))
        if self.with_dropout:
            x = self.dropout(x)
        x = F.relu(self.mlp4_bn2_2(self.mlp4_conv4(x)))
        
        # output
        x = self.output_conv(x)
        x = x.transpose(2,1).contiguous()
        x = torch.nn.Softmax(dim=-1)(x.view(-1, self.num_classes))
        x = x.view(batchsize, n_pts, self.num_classes)

        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeshSegNet_L().to(device)
    summary_(model, [(15, 6000), (6000, 6000), (6000, 6000)])
