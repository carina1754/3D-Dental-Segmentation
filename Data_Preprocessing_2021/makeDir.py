import os
path = os.path.dirname(os.path.abspath(__file__))
rootpath = path + '/data'
datapath = path + '/segmentation data'
import time
start = time.time()


os.makedirs(rootpath, exist_ok=True)
os.makedirs(rootpath+'/labels', exist_ok=True)
os.makedirs(rootpath+'/points', exist_ok=True)

segment_data_name = '' 
segment_data_name_list = list()
part_data_name = ''
part_data_name_list = list()
name = ""
result_label_list = list()
result_point_list = list()

def all_point(file_name):
    point = 0
    with open(file_name,'r') as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == 'v':
            if line[1] == 'n':
                continue
            else: point += 1
        else: return point



def main(): # 시작
    data_list = os.listdir(datapath)
    for data in data_list:
        if data[0] == 'd':
            segment_data_name_list.append(data)
        elif data[0] == 't':
            part_data_name_list.append(data)
    
    for segment_data in segment_data_name_list: #segment_data 형식은 d_0.obj, d_1.obj, d_2,obj...
        segment_name = segment_data.split('.')
        point = 1
        nice_point = 0
        result_list = list()
        print(segment_data)
        with open(datapath + '/' + segment_data,'r') as segfile:
            segment_lines = segfile.readlines()
        for segment_line in segment_lines: #d_0.obj, d_1.obj, d_2,obj.. 를 한 줄 씩 읽는다.
            flag = True
            result_name = ''
            segment_line_split = segment_line.split()
            
            if segment_line_split[0] == 'vn': # vn이면 바로 런
                continue
            if segment_line_split[0] != 'v': # v 다 보면 break
                break
            
            # 탐색 알고리즘
            for part_data in part_data_name_list: #part_data를 불러온다. part_data 형식은 t_11_0.obj, t_12_0.obj,... 
                #print(part_data)
                with open(datapath + '/' + part_data, 'r') as partfile:
                    part_lines = partfile.readlines()
                    for part_line in part_lines: 
                        
                        part_line_split = part_line.split()
                        if part_line_split[0] == 'vn':
                            continue
                        if part_line_split[0] != 'v':
                            break
                        
                        if segment_line_split[1] == part_line_split[1] and segment_line_split[2] == part_line_split[2] and segment_line_split[3] == part_line_split[3]: #일치한다면, 
                            nice_point += 1
                            # print(segment_line_split[1] + ' ' + part_line_split[1] + ' / ' + segment_line_split[2] + ' ' + part_line_split[2] + ' / ' + segment_line_split[3] + ' ' + part_line_split[3])
                            flag = False
                            part_name = part_data.split('_')
                            result_list.append(part_name[1])
            if point % 200 == 0:
                end = time.time()
                print(end - start)
                print(str(point) + ' ' + str(nice_point))
           
            if flag == True:
                result_list.append('0')
            point += 1
        name = rootpath+'/labels/'+segment_name[0] + '.seg'
        with open(name,'w') as f:
            for result in result_list:
                f.write(result + '\n')
                        
    return 

"""
def print_file(dir):
    file_list = os.listdir(dir)

    for file in file_list:
        if os.path.isdir(dir+r"\\"+file) == True :
            print_file(dir+r"\\"+file)
            text = file[len(file)-4:]
            if text == '.off':
                print(file)
        else:
            print(dir+r"\\"+file)

dir = 'C:\Download\example'

print_file(dir)
"""

if __name__ == "__main__":
   main()