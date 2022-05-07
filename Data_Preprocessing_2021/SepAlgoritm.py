import os
path = os.path.dirname(os.path.abspath(__file__))
rootpath = path + '/Newdata'
datapath = path + '/NewSegmentationData'
import time
start = time.time()
segment_all_data = dict()
    # segment_all_data = 'd_000':['x1 y1 z1',
    #                             'x2 y2 z2',
    #                                 ...   ]
result_all_seg = dict()

def load_segment_data(file_name): # file_name은 d_000.obj.. 이런 애들
    segment_candidate_data = list()
    name = list()
    dict_name = file_name[:-4]
    name = dict_name.split('_')
    real_name = name[1] + '_' + name[2]
    segment_all_data[real_name] = list()
    with open(file_name, 'r') as f:
        seg_lines = f.readlines()
    for line in seg_lines:
        line_split = line.split()
        if line_split[0] == 'vn':
            continue
        if line_split[0] !=  'v':
            break
        name = line_split[0] + ' ' + line_split[1] + ' ' +line_split[2] + ' ' + line_split[3]
        segment_candidate_data.append(name)
    segment_all_data[real_name] = segment_candidate_data
    global segment_data_name_list
    segment_data_name_list = list(segment_all_data.keys()) # ['d_000', 'd_001', ...]

def init_result_data(ordered_segmentation_data): # result_all_seg 딕셔너리를 모두 0으로 초기화 하는 것.
    for key in list(ordered_segmentation_data.keys()):
        result_all_seg[key] = list()
        for i in range(len(ordered_segmentation_data[key])):
            result_all_seg[key].append('0')

def new_binary_search(list, element, start_idx, end_idx, key, label): # list는 ['x1 y1 z1', x2 y2 z2', 'x3 y3 z3', ...] 이런 형식이다.
    mid_idx = (start_idx + end_idx)//2
    for idx in range(start_idx, mid_idx):
        if list[idx] == element:
            result_all_seg[key][idx] = label
            return True
    if start_idx == end_idx - 1: #다 찾았는데 없으면, 그 리스트는 False return
        return False

    return new_binary_search(list, element, mid_idx, end_idx, key, label)

def new_searching(data):
    data_name_path = datapath + '/' + data
    data_name = data[:-4]
    data_name = data_name.split('_')
    data_label = data_name[3]
    data_name = data_name[1] + '_' + data_name[2]
    with open(data_name_path, 'r') as f:
        par_lines = f.readlines()
    #number = 1
    for line in par_lines:
        line_split = line.split()
        if line_split[0] == 'vn':
            continue
        if line_split[0] != 'v':
            break
        #if number % 100 == 0:
        #    print(str(number) + '개 분류 완료.')
        point = line_split[0] + ' ' + line_split[1] + ' ' +line_split[2] + ' ' + line_split[3]
        new_binary_search(segment_all_data[data_name], point, 0, len(segment_all_data[data_name]) + 1, data_name, data_label)
        #number += 1

def main():

    os.makedirs(rootpath, exist_ok=True)
    os.makedirs(rootpath+'/labels', exist_ok=True)
    os.makedirs(rootpath+'/points', exist_ok=True)


    ########## segment_all_data 와 part_all_data 정리 ##########
    data_list = os.listdir(datapath)
    for data in data_list:
        if data[0] == 'd':
            load_segment_data(datapath + '/' + data)
    ############################################################
    print('기초 load 완료')

    ########## result_all_data 정리 ##########
    init_result_data(segment_all_data)
    ##########################################
    print('init 완료')

    for data in data_list:
        if data[0] == 't':
            new_searching(data)
            print(data + ' serching complete')
    ###############################

    ########## saving ##########
    for key in list(result_all_seg.keys()):
        with open(rootpath+'/labels/'+key+'.seg', 'w') as f:
            for data in result_all_seg[key]:
                f.write(data + '\n')
        f.close()
    for key in list(result_all_seg.keys()):
        with open(rootpath+'/points/'+key+'.pts','w') as f:
            for data in segment_all_data[key]:
                line_split = data.split()
                line = line_split[1] + ' ' + line_split[2] + ' ' + line_split[3]
                f.write(line + '\n')
        f.close()
    ############################
    print('saving 완료')

    return

if __name__ == "__main__":
	main()
