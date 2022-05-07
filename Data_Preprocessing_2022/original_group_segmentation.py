import os
import re
import json
import shutil
from tqdm import tqdm
path = "C:\\Users\\buleb\\Desktop\\3D-Semantic-Segmentation"
# rootpath = path + '\\datas\\segmentation-files(2)'
rootpath = 'original\\' ## output if saved in the root path
labelpath = rootpath + '\\labels'
fullpath = rootpath + '\\fullScans'
datapath = path + '\\labels_restore' ### this is the input data path
temppath = 'tmp/'
import time
start = time.time()

def fufu(dir, output_dir):
    
    vertex_ = list()
    faces_ = list()
    face_dummy = list()

    os.makedirs(output_dir, exist_ok=True)

    with open(dir, 'r') as f:
        lines = f.readlines()

    for line in lines : # fill vertex
        line_split = line.split()
        if line_split[0] == 'v':
            vertex_.append(line)
    
    temp_group_dummy = False
    face_state = False
    face_id = 0
    face_cand = list()
    for line in lines : # fill face
        line_split = line.split()
        if line_split[0] == 'g':
            l = line

            face_id += 1
            if face_state == True:
                if temp_group_dummy == False:
                    faces_.append(face_cand)
                elif temp_group_dummy == True:
                    face_dummy.append(face_cand)

            face_cand = list()
            face_cand.append('# mm_gid 0\n')
            face_cand.append(l)
            if line == 'g mmGroup0\n':
                temp_group_dummy = True
            else:
                temp_group_dummy = False
        elif line_split[0] == 'f':
            face1 = line_split[1].split('//')[0]
            face2 = line_split[2].split('//')[0]
            face3 = line_split[3].split('//')[0]
            face_cand.append('f {a}//{a} {b}//{b} {c}//{c}'.format(a=face1,b=face2,c=face3)+'\n')
            face_state = True

    if temp_group_dummy == True:
        face_dummy.append(face_cand)
    else:
        faces_.append(face_cand)
    
    write_line = list()
    for ver in vertex_:
        write_line.append(ver)

    for dummy in face_dummy:
        for d in dummy:
            write_line.append(d)
    
    for face in faces_:
        for f in face:
            write_line.append(f)

    name = 'Sample_0.obj'
    with open(output_dir + '/' + name,'w') as out:
        for w in write_line:
            out.write(w)
    
    for line in write_line:
        line_split = line.split()

    vertex = list()
    faces = list()
    ver_ind = 0

    face_state = False
    ss = 0
    lines = write_line
    for line in lines:
        line_split = line.split()
        if ver_ind == 0:
            if line_split[0] == 'v':
                vertex.append(line)
            if line_split[0] == '#':
                ver_ind = 1
        else:
            if line_split[0] == 'g':
                if face_state == False:
                    face_cand = list()
                    face_cand.append('# mm_gid 0\n')
                    face_cand.append(line)
            elif line_split[0] == 'f':
                face_state = True
                face_cand.append(line)
            elif line_split[0] == '#':
                face_state = False
                faces.append(face_cand)

    faces.append(face_cand)

    fac_ind = 0
    removeflag = False
    for face in faces:
        if fac_ind == 0:
            fac_ind += 1
            continue
        else:
            if removeflag:
                removeflag = False
                break
            name = 'Sample_' + str(fac_ind) + '.obj'
            with open(temppath + name,'w') as out:
                for v in vertex:
                    ver = v.split(" ")
                    out.write(ver[0] + ' ' + ver[1] + ' ' + ver[2] + ' ' + ver[3]+'\n')
                for fa in face:
                    faces = fa.split(' ')
                    if faces[0] == 'f':
                        face1 = faces[1].split('//')[0]
                        face2 = faces[2].split('//')[0]
                        face3 = faces[3].split('//')[0]
                        out.write('f {a}//{a} {b}//{b} {c}//{c}'.format(a=face1,b=face2,c=face3)+'\n')
                    else:
                        if fa.startswith('g'):
                            group = fa
                            out.write(group)
                        else:
                            out.write(fa)
            shutil.copy(temppath + name,output_dir + '/' + 'Sample_' + str(group[9:-1]) + '.obj')
            fac_ind += 1 
    # os.remove(output_dir+'/'+'Sample_' + str(fac_ind-1) + '.obj')
    if fac_ind > 18:
        for i in os.listdir(output_dir):
            if os.path.isfile(output_dir+'/'+i):
                os.remove(output_dir+'/'+i)
        os.rmdir(output_dir)
        removeflag = True
    shutil.rmtree(temppath+'/')
    os.mkdir(temppath)

def func(dir):
    file_list = os.listdir(dir)
    for file in tqdm(file_list):
        # print(rootpath +r"\\"+file)
        # fufu(datapath + r"\\" + file, rootpath + r"\\" +file)
        if file[-9:-8] == 'U':
            # print(file)
            output_dir = rootpath + '\\upper\\' + file[:-8]
            fufu(datapath + r"\\" + file, output_dir)
        elif file[-9:-8] == 'L':
        #    print(file)
            output_dir = rootpath + '\\lower\\' + file[:-8]
            fufu(datapath + r"\\" + file, output_dir)
func(datapath)