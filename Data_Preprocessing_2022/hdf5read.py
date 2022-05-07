import hdfdict
import pandas as pd

res = hdfdict.load('h5data(6000)_one/Sample_01_d.h5')
resdict = dict(res)

# data_list = pd.read_csv('train_list_1.csv', header=None)
# i_mesh = data_list.iloc[1][0]
# res = hdfdict.load(i_mesh)

for i in dict(res)['cells']:
    print(i)