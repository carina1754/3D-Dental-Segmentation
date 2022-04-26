from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix
import hdfdict
class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path):
        """
        Args:
            h5_path (string): Path to the txt file with h5 files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = pd.read_csv(data_list_path, header=None)
    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        i_mesh = self.data_list.iloc[idx][0] #vtk file name
        # print(i_mesh)
        res = hdfdict.load(i_mesh)
        return dict(res)
