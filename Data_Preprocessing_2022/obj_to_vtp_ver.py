from vedo import *
import os
from tqdm import tqdm
if __name__ == '__main__':
    # for i in os.listdir('seg_data(1)/lower'):
    #     if os.path.isfile(f'./result_vtp/lower/{i}.vtp'):
    #         continue
    #     elif i.startswith('.'):
    #         continue
    #     else:
    #     # load obj file
    #         data  = []
    #         mesh = load('seg_data(1)/lower/' + i + '/Sample_0.obj')
    #         obj = open('seg_data(1)/lower/' + i + '/Sample_0.obj','r')
    #         lines = obj.readlines()
    #         for line in lines:
    #             if line.startswith('g'):
    #                 data.append(round(float(line[9:-1]),1))    
    #         # get the cell array namely 'GroupIds'
    #         label_array = mesh.celldata['GroupIds']
    #         for label in range(len(label_array)):
    #             if label_array[label] == 0.0:
    #                 label_array[label] = 23
    #             else: 
    #                 index = int(label_array[label])
    #                 label_array[label] = data[index]
    #         # assign this cell array to a new cell array called 'Label', which is used in Mesh Labeler
    #         mesh.celldata['Label'] = label_array # note this one is not working in latest vedo; my version is 2021.0.3]
    #         # delete un-used array (option)
    #         mesh.polydata().GetCellData().RemoveArray('GroupIds')
    #         mesh.polydata().GetPointData().RemoveArray('Normals')
    #         # write the mesh object to VTP format
    #         write(mesh, f'./result_vtp/lower/{i}.vtp')
    for i in os.listdir('seg_data(1)/upper'):
        print(i)
        if os.path.isfile(f'./result_vtp/upper/{i}.vtp'):
            continue
        elif i.startswith('.'):
            continue
        else:
            data  = []
            mesh = load('seg_data(1)/upper/' + i + '/Sample_0.obj')
            obj = open('seg_data(1)/upper/' + i + '/Sample_0.obj','r')
            lines = obj.readlines()
            for line in lines:
                if line.startswith('g'):
                    data.append(round(float(line[9:-1]),1))    
            # get the cell array namely 'GroupIds'
            label_array = mesh.celldata['GroupIds']
            for label in range(len(label_array)):
                if label_array[label] == 0.0:
                    label_array[label] = 23
                else: 
                    index = int(label_array[label])
                    label_array[label] = data[index]
            # assign this cell array to a new cell array called 'Label', which is used in Mesh Labeler
            mesh.celldata['Label'] = label_array # note this one is not working in latest vedo; my version is 2021.0.3
            # delete un-used array (option)
            mesh.polydata().GetCellData().RemoveArray('GroupIds')
            mesh.polydata().GetPointData().RemoveArray('Normals')
            # write the mesh object to VTP format
            write(mesh, f'./result_vtp/upper/{i}.vtp')
