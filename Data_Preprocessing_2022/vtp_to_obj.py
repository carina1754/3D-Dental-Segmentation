from esay_mesh_vtk import Easy_Mesh
import random
import os
import re
import sys
def main(vtpname,resultname):
    # output as wavefront (.obj) that can be visualized colorfully by MeshMixer with face groups
    mesh = Easy_Mesh(vtpname)
    mesh.to_obj('Sample_')

    path = os.path.dirname(os.path.abspath(__file__))
    rootpath = path + '/vtptoobj' ## 여기에서 data input
    datapath = path+'/result/'+resultname ## 새로운 데이터

    vertex_list = list()
    vertex_list_check = list()
    vertex_dir_list = list()
    face_list = list()
    face_dir_list = list()
    color = [[10,0,10],
            [10,10,0],
            [4,10,10],
            [10,10,4],
            [8,10,10],
            [10,8,10],
            [10,8,6],
            [10,6,8],
            [8,10,6],
            [8,6,10],
            [6,8,10],
            [10,8,2],
            [10,4,4],
            [4,10,4],
            [4,4,10]]

    def check_dir(dir):
        file_list = os.listdir(dir)

        for file in file_list:        
            if os.path.isdir(dir+r"\\"+file) == False :
                if file[-4:] == '.obj':
                    split_file = file.split('.')
                    if  len(split_file) == 2: # 데이터 형식이 Sample_7.obj 처럼 . 으로 2개로 split  되어야함
                        face_dir_list.append(dir+r"\\"+file)
                    else:
                        vertex_dir_list.append(dir+r"\\"+file)

    check_dir(rootpath)

    with open(face_dir_list[0],'r') as f:
        lines = f.readlines()

    for line in lines:
        if line[0] == 'v':
            vertex_list.append(line)
            vertex_list_check.append(0)
        else:
            break

    color_ind = -1
    for file in face_dir_list:
        color_ind += 1
        r = random.randint(0,10) /10
        r = color[color_ind][0] / 10
        g = random.randint(0,10)  / 10
        g = color[color_ind][1]  / 10
        b = random.randint(0,10)  / 10
        b = color[color_ind][2]  / 10
        with open(file,'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line[0] == 'v':
                continue
            elif line[0] == 'g':
                face_list.append('# mm_gid 0\n')
                face_list.append(line)
            elif line[0] == 'f':
                face_list.append(line)
                split_line = line.split()
                f_l = list()
                first = split_line[1].split('//')
                first = int(first[0])
                f_l.append(first)
                sec = split_line[2].split('//')
                sec = int(sec[0])
                f_l.append(sec)
                thr = split_line[3].split('//')
                thr = int(thr[0])
                f_l.append(thr)
                for f in f_l:
                    if vertex_list_check[f - 1] == 0:
                        vertex_list[f - 1] = vertex_list[f - 1].strip()
                        vertex_list[f - 1] = vertex_list[f - 1] + ' ' + str(r) + '00000 ' + str(g) + '00000 ' + str(b) + '00000\n'
                        vertex_list_check[f - 1] = 1
                    else:
                        continue

    with open(datapath, 'w') as out:
        ind = 0
        for vertex in vertex_list:
            ver_split = vertex.split()
            num = float(ver_split[1])
            ver_split[1] = str(round(num,6))
            num = float(ver_split[2])
            ver_split[2] = str(round(num,6))
            num = float(ver_split[3])
            ver_split[3] = str(round(num,6))
            out.write(ver_split[0] + ' ' + ver_split[1]+ ' ' + ver_split[2]+ ' ' + ver_split[3]+ ' ' + ver_split[4]+ ' ' + ver_split[5]+ ' ' + ver_split[6] + '\n')

        for face in face_list:
            #if face_list_check[ind] == 1:
            #    ind+=1
            #    continue
            out.write(face)
            ind+=1




if __name__ == '__main__':
    main()
    print('sdfsdf')