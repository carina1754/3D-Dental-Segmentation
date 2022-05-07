import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix
import torch.nn as nn
import math
import time
import h5py
from tqdm import tqdm
def makeh5(data,h5_path):
    if os.path.isfile(h5_path+data[25:]+'.h5'):
        print('1' + data)
        
    else:
        i_mesh = data+'.vtp' #vtk file name

        # read vtk
        mesh = load(i_mesh)
        labels = mesh.celldata['Label'].astype('int32').reshape(-1, 1)

        # move mesh to origin
        cells = np.zeros([mesh.NCells(), 9], dtype='float32')
        for i in range(len(cells)):
            cells[i][0], cells[i][1], cells[i][2] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(0)) # don't need to copy
            cells[i][3], cells[i][4], cells[i][5] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(1)) # don't need to copy
            cells[i][6], cells[i][7], cells[i][8] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(2)) # don't need to copy

        mean_cell_centers = mesh.centerOfMass()
        cells[:, 0:3] -= mean_cell_centers[0:3]
        cells[:, 3:6] -= mean_cell_centers[0:3]
        cells[:, 6:9] -= mean_cell_centers[0:3]

        # customized normal calculation; the vtk/vedo build-in function will change number of points
        v1 = np.zeros([mesh.NCells(), 3], dtype='float32')
        v2 = np.zeros([mesh.NCells(), 3], dtype='float32')
        v1[:, 0] = cells[:, 0] - cells[:, 3]
        v1[:, 1] = cells[:, 1] - cells[:, 4]
        v1[:, 2] = cells[:, 2] - cells[:, 5]
        v2[:, 0] = cells[:, 3] - cells[:, 6]
        v2[:, 1] = cells[:, 4] - cells[:, 7]
        v2[:, 2] = cells[:, 5] - cells[:, 8]
        mesh_normals = np.cross(v1, v2)
        mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
        mesh_normals[:, 0] /= mesh_normal_length[:]
        mesh_normals[:, 1] /= mesh_normal_length[:]
        mesh_normals[:, 2] /= mesh_normal_length[:]
        mesh.celldata['Normal'] = mesh_normals

        # preprae input and make copies of original data
        points = mesh.points().copy()
        points[:, 0:3] -= mean_cell_centers[0:3]
        normals = mesh.celldata['Normal'].copy() # need to copy, they use the same memory address
        barycenters = mesh.cellCenters() # don't need to copy
        barycenters -= mean_cell_centers[0:3]

        #normalized data
        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
            cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
            cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
            barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
            normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

        X = np.column_stack((cells, barycenters, normals))
        Y = labels

        X_train = np.zeros([6000, X.shape[1]], dtype='float32')
        Y_train = np.zeros([6000, Y.shape[1]], dtype='int32')
        S1 = np.zeros([6000, 6000], dtype='float32')
        S2 = np.zeros([6000, 6000], dtype='float32')
        S3 = np.zeros([6000, 6000], dtype='float32')
        
        positive_idx = np.argwhere(labels>0)[:, 0] #tooth idxp
        
        positive_selected_idx = np.random.choice(positive_idx, size=6000, replace=False)
        selected_idx = positive_selected_idx
        
        selected_idx = np.sort(selected_idx, axis=None)
        X_train[:] = X[selected_idx, :]
        Y_train[:] = Y[selected_idx, :]
        
        D = distance_matrix(X_train[:, 9:12], X_train[:, 9:12])
        
        S1[D<0.1] = 1.0
        S1 = S1 / np.dot(np.sum(S1, axis=1, keepdims=True), np.ones((1, 6000)))

        S2[D<0.2] = 1.0
        S2 = S2 / np.dot(np.sum(S2, axis=1, keepdims=True), np.ones((1, 6000)))
        
        S3[D<0.3] = 1.0
        S3 = S3 / np.dot(np.sum(S3, axis=1, keepdims=True), np.ones((1, 6000)))

        X_train = X_train.transpose(1, 0)
        Y_train = Y_train.transpose(1, 0)
        labels = torch.from_numpy(Y_train)
        one_hot_labels = nn.functional.one_hot(labels.to(dtype=torch.long), num_classes=24)
        f = h5py.File(h5_path+i_mesh[25:-3]+'h5','w')
        f.create_dataset('cells', data=torch.from_numpy(X_train))
        f.create_dataset('labels', data=one_hot_labels)
        f.create_dataset('A_S', data=torch.from_numpy(S1))
        f.create_dataset('A_L', data=torch.from_numpy(S2))
        f.create_dataset('A_U', data=torch.from_numpy(S3))
        f.close()
    
h5_path='data\\'
data_list_path = 'tensor_list\\val_list_1_vtp_1000.csv'
data_list = pd.read_csv(data_list_path, header=None)
for i in tqdm(data_list[0]):
    makeh5(i,h5_path)