from os.path import getsize
import pandas as pd
data_list_path = 'tensor_list/val_list_1_vtp.csv'
#file2 = 'D:\test.zip'
data_list = pd.read_csv(data_list_path, header=None)
redata_list = []
for i in data_list[0]:
    file_size = getsize(i+'.vtp')
    if file_size < 1000000:
        continue
    else:
        redata_list.append(i)
df = pd.DataFrame(redata_list)
df.to_csv('val_list_1_vtp_1000.csv',index=False)
# print('File Nama: %s \tFile Size: %d' %(file1, file_size1))
# #print('File Name: %s \tFile Size: %d' %(file2, file_size2))1021011