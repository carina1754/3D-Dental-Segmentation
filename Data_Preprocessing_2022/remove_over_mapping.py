import os
import shutil
from tqdm import tqdm
datapath = "blender2"
def countfile(dir):
    file_list = os.listdir(dir)
    if len(file_list)>17:
        os.rmdir(dir)
        print(dir)
    else:
        for file in file_list:
            if 'None' in file:
                for file in file_list:
                    os.remove(dir+'\\'+file)
                os.rmdir(dir)
                print(dir)
file_list = os.listdir(datapath + '\\upper\\')
for file in tqdm(file_list):
    data = datapath + '\\upper\\' + file
    countfile(data)
file_list = os.listdir(datapath + '\\lower\\')
for file in tqdm(file_list):
    data = datapath + '\\lower\\' + file
    countfile(data)