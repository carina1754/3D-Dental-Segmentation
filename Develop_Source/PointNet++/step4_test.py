import os
import numpy as np
import torch
import torch.nn as nn
from pointnet import *
import utils
from easy_mesh_vtk import *
import pandas as pd
from losses_and_metrics_for_mesh import *

if __name__ == '__main__':
    
    torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)
      
    model_path = './models'
    #model_name = 'checkpoint.tar'
    model_name = 'Mesh_Segementation_PointNet_15_classes_72samples_best.tar'
    
    mesh_path = '/home/brucewu/data/OSU_Mesh_Tooth_Segmentation/GroundTruth/maxillary_surface_T0_np10000_bg_0_vtk'
    test_list = [31, 32, 33, 34, 35, 36]
    test_mesh_filename = 'Sample_0{0}_d.vtp'
    test_path = './test'
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    
    num_classes = 15
    num_features = 15
          
    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet_Seg(num_classes=num_classes, channel=num_features).to(device, dtype=torch.float)
    
    # load trained model
    checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model = model.to(device, dtype=torch.float)
    
    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Testing
    dsc = []
    sen = []
    ppv = []
    
    print('Testing')
    model.eval()
    with torch.no_grad():
        for i_sample in test_list:
            
            print('Predicting Sample filename: {}'.format(test_mesh_filename.format(i_sample)))
            # read image and label (annotation)
            mesh = Easy_Mesh(os.path.join(mesh_path, test_mesh_filename.format(i_sample)))
            labels = mesh.cell_attributes['Label'].astype('int32')
            predicted_labels = np.zeros(labels.shape)
        
            # move mesh to origin
            cell_centers = (mesh.cells[:, 0:3] + mesh.cells[:, 3:6] + mesh.cells[:, 6:9])/3.0
            mean_cell_centers = np.mean(cell_centers, axis=0)
            mesh.cells[:, 0:3] -= mean_cell_centers[0:3]
            mesh.cells[:, 3:6] -= mean_cell_centers[0:3]
            mesh.cells[:, 6:9] -= mean_cell_centers[0:3]
            mesh.update_cell_ids_and_points() # update object when change cells
            mesh.get_cell_normals() # get cell normal

            # preprae input
            cells = mesh.cells[:]
            normals = mesh.cell_attributes['Normal'][:]
            cell_ids = mesh.cell_ids[:]
            points = mesh.points[:]
            barycenters = (cells[:,0:3]+cells[:,3:6]+cells[:,6:9]) / 3

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
            X = (X-np.ones((X.shape[0], 1))*np.mean(X, axis=0)) / (np.ones((X.shape[0], 1))*np.std(X, axis=0))
            X = X.transpose(1, 0)
            X = X.reshape([1, X.shape[0], X.shape[1]])
            
            # numpy -> torch.tensor
            X = torch.from_numpy(X).to(device, dtype=torch.float)
            
            tensor_prob_output = model(X).to(device, dtype=torch.float).detach()
            patch_prob_output = tensor_prob_output.cpu().numpy()
                
            for i_label in range(num_classes):
                predicted_labels[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

            # output predicted labels
            mesh = Easy_Mesh(os.path.join(mesh_path, test_mesh_filename.format(i_sample)))
            mesh2 = Easy_Mesh()
            mesh2.cells = mesh.cells
            mesh2.update_cell_ids_and_points()
            mesh2.cell_attributes['Label'] = predicted_labels
            mesh2.to_vtp(os.path.join(test_path, 'Sample_{}_deployed.vtp'.format(i_sample)))

            # convert predict result and label to one-hot maps
            tensor_predicted_labels = torch.from_numpy(predicted_labels)
            tensor_test_labels = torch.from_numpy(labels)
            tensor_predicted_labels = tensor_predicted_labels.long()
            tensor_test_labels = tensor_test_labels.long()
            one_hot_predicted_labels = nn.functional.one_hot(tensor_predicted_labels[:, 0], num_classes=num_classes)
            one_hot_labels = nn.functional.one_hot(tensor_test_labels[:, 0], num_classes=num_classes)
            
            # calculate DSC
            i_dsc = DSC(one_hot_predicted_labels, one_hot_labels)
            i_sen = SEN(one_hot_predicted_labels, one_hot_labels)
            i_ppv = PPV(one_hot_predicted_labels, one_hot_labels)
            dsc.append(i_dsc)
            sen.append(i_sen)
            ppv.append(i_ppv)
            #print('\tLabel 1: {}; Label 2: {}'.format(dsc[0], dsc[1]))
        
    dsc = np.asarray(dsc)
    sen = np.asarray(sen)
    ppv = np.asarray(ppv)

    # output all DSCs
    all_dsc = pd.DataFrame(data=dsc, index=test_list, columns=['label {}'.format(i) for i in range(1, num_classes)])
    all_sen = pd.DataFrame(data=sen, index=test_list, columns=['label {}'.format(i) for i in range(1, num_classes)])
    all_ppv = pd.DataFrame(data=ppv, index=test_list, columns=['label {}'.format(i) for i in range(1, num_classes)])
    print(all_dsc)
    print(all_dsc.describe())
    print(all_sen)
    print(all_sen.describe())
    print(all_ppv)
    print(all_ppv.describe())
    all_dsc.to_csv(os.path.join(test_path, 'test_DSC_report.csv'), header=True, index=True)
    all_sen.to_csv(os.path.join(test_path, 'test_SEN_report.csv'), header=True, index=True)
    all_ppv.to_csv(os.path.join(test_path, 'test_PPV_report.csv'), header=True, index=True)
            
