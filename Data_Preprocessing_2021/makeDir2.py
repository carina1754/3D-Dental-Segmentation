import os
path = os.path.dirname(os.path.abspath(__file__))
rootpath = path + '/data'
datapath = path + '/ex'
import time
start = time.time()

segment_data_name = '' 
part_data_name = ''
part_data_name_list = list()
name = ""
result_label_list = list()
result_point_list = list()



segment_all_data = dict()
    # segment_all_data = 'd_000':['x1 y1 z1',
    #                             'x2 y2 z2',
    #                                 ...   ]
part_all_data = dict()
    # part_all_data = '11':[ 'x1 y1 z1',
    #                        'x2 y2 z2',
    #                            ...    ]
result_all_seg = dict()
result_all_pts = dict()
seg_data_num = 0
par_data_num = 0

def load_segment_data(file_name): # file_name은 d_000.obj.. 이런 애들
    segment_candidate_data = list()
    name = list()
    dict_name = file_name.split('.')
    name = dict_name[0].split('_')
    real_name = name[0][-1:] + '_' + name[1]
    print(real_name)
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


def load_part_data(file_name): # file_name은 t_11_1.obj.. 이런 애들
    part_candidate_data = list() 
    name = list()
    flag = True # 이미 key가 존재 하는지, 안하는지. 새로운 key일 때 true
    dict_name = file_name.split('_')
    for key in list(part_all_data.keys()):
        if key == dict_name[1]:
            flag = False
    if flag: # 새로운 key이면 새로 리스트를 만듬
        part_all_data[dict_name[1]] = list()
    with open(file_name, 'r') as f:
        par_lines = f.readlines()
    for line in par_lines:
        line_split = line.split()
        if line_split[0] == 'vn':
            continue
        if line_split[0] != 'v':
            break
        name = line_split[0] + ' ' + line_split[1] + ' ' +line_split[2] + ' ' + line_split[3]
        part_candidate_data.append(name)

    if flag: #새로운 key이면 part_candidate_data 가 통째로 들어옴
        part_all_data[dict_name[1]] = part_candidate_data
    else : # 이미 존재하는 key면 추가만 해줌
        for data in part_candidate_data:
            part_all_data[dict_name[1]].append(data)

    global dict_index_list
    dict_index_list = list(part_all_data.keys()) # ['11'(key), '12'(key), '14'(key), ...]

def init_result_data(ordered_segmentation_data): # result_all_seg 딕셔너리를 모두 0으로 초기화 하는 것.
    for key in list(ordered_segmentation_data.keys()):
        result_all_seg[key] = list()
        for i in range(len(ordered_segmentation_data[key])):
            result_all_seg[key].append('0')

        


def binary_search(list, element, start_idx, end_idx): # list는 ['x1 y1 z1', x2 y2 z2', 'x3 y3 z3', ...] 이런 형식이다.
    mid_idx = (start_idx + end_idx)//2
    for idx in range(start_idx, mid_idx):
        if list[idx] == element:
            global global_Xinx 
            global_Xinx = idx
            return True
    if start_idx == end_idx - 1: #다 찾았는데 없으면, 그 리스트는 False return
        return False

    return binary_search(list, element, mid_idx, end_idx)

def search(element, start_idx, end_idx): # segment_all_data를 2개씩 나누어가면서 찾는다. 라벨이 d_001 d_002 d_003 d_004 이라고하면, (d_001 d_002) 를 찾고 없으면 d_003 d_004 찾아가기.
    mid_idx = (start_idx + end_idx) // 2
    for idx in range(start_idx, mid_idx):
        if binary_search(segment_all_data[segment_data_name_list[idx]], element, 0, len(segment_all_data[segment_data_name_list[idx]]) + 1):
            
            global global_Yinx
            global_Yinx = idx
            return True
        else: search(element, mid_idx, end_idx)
    return False

def main():

    os.makedirs(rootpath, exist_ok=True)
    os.makedirs(rootpath+'/labels', exist_ok=True)
    os.makedirs(rootpath+'/points', exist_ok=True)


    ########## segment_all_data 와 part_all_data 정리 ##########
    data_list = os.listdir(datapath)
    for data in data_list:
        if data[0] == 'd':
            load_segment_data(datapath + '/' + data)
        if data[0] == 't':
            load_part_data(datapath + '/' + data)
    ############################################################
    print('기초 load 완료')

    ########## result_all_data 정리 ##########
    init_result_data(segment_all_data)
    ##########################################
    print('init 완료')

    seg_data_num = len(list(segment_all_data.keys()))

    ########## searching ##########
    for key in list(part_all_data.keys()):
        number = 1
        print(len(part_all_data[key]))
        for point in part_all_data[key]:
            if number % 100 == 0:
                print(str(key) + '에서 ' + str(number) +'개 찾음')
            search(point, 0, seg_data_num + 1)
            result_all_seg[segment_data_name_list[global_Yinx]][global_Xinx] = key
            number += 1
        print(str(key) + '찾음')
    ###############################
    print('searching 완료')

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
