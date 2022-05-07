import os
path = os.path.dirname(os.path.abspath(__file__))
rootpath = path + '/data'
pathlist = list()
pathlist.append(rootpath)
os.makedirs(rootpath, exist_ok=True)

dir = 'E:/2021ProjectPro/pointnet.pytorch-master/pointnet.pytorch-master/dataset/ModelNetDataset/ModelNet40'

# 'dir'은 off로 저장이 되어 있는 파일의 위치를 가르킨다.
#  search_file(dir) 를 입력하면 자동으로 data라는 폴더를 만들고,
#  그 폴더 안에 off 저장 파일의 모양과 똑같은 형태로 ply파일을 만든다.
# 'dir'에 변경시키고싶은 파일 경로만 집어넣으면 됨.

def off_to_ply(filename,copy_filename):
    write_down = list()
    with open(filename, 'r') as f:
        lines = f.readlines()
    split_line = lines[1].split()
    vertexs = split_line[0]
    faces = split_line[1]
    write_down.append('ply\n')
    write_down.append('format ascii 1.0\n')
    write_down.append('element vertex ' + vertexs + '\n')
    write_down.append('property float32 x\n')
    write_down.append('property float32 y\n')
    write_down.append('property float32 z\n')
    write_down.append('element face ' + faces + '\n')
    write_down.append('property list uint8 int32 vertex_indices\n')
    write_down.append('end_header\n')
    index = 0
    for line in lines:
        if index == 0 or index == 1:
            index = index + 1
            continue
        write_down.append(line)
        index = index + 1
    f.close()

    output_name = copy_filename[:-4]
    output_name = output_name + '.ply'
    with open(output_name,'w') as f:
        for line in write_down:
            f.write(line)
    f.close()

def search_file(dir):
    file_list = os.listdir(dir)

    for file in file_list:
        if os.path.isdir(dir+r"\\"+file) == True : #폴더면,
            pathlist.append(pathlist[-1]+r"\\"+file)
            os.makedirs(pathlist[-1], exist_ok=True)

            search_file(dir+r"\\"+file)
            pathlist.pop()
        else:
            text = file[len(file)-4:]
            if text == '.off':
                off_to_ply(dir+r"\\"+file,pathlist[-1]+r"\\"+file)
                #print(file)
           
search_file(dir)