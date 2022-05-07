import os
import json
import shutil
from vedo import *
path = "C:\\Users\\buleb\\Desktop\\3D-Semantic-Segmentation"
rootpath = path + '\\Labels_and_offs2'
labelpath = rootpath + '\\labes'
datapath = path + '\\datafile' # input
returnpath = path + '\\labels_restore'# output
resultpath = 'result' ## output if saved in the root path
temppath = 'tmp/'
original_file_list = list() 
original_file_name_list = list() 
copy_file_list = list() 
copy_file_check = list() 


def init_dirs(dir):
    file_list = os.listdir(dir)

    for file in file_list:
        if os.path.isdir(dir+r"\\"+file) == True :
            original_file_list.append(dir+r"\\"+file)
            original_file_name_list.append(file)
            copy_file_check.append(file)
            input_name = ""
            for put_name in copy_file_check:
                input_name += (r"\\"+put_name)
            copy_file_list.append(rootpath+input_name)                                                                               
            init_dirs(dir+r"\\"+file)
    if len(copy_file_check) != 0:
        copy_file_check.pop()

def obj2off(objpath, offpath, offpath2):
    with open(objpath,'r') as f:
        lines = f.readlines()
       
    with open(offpath, 'w') as out:
        out = open(offpath, 'w')
        for line in lines:
            out.write(line + " 0\n")

    with open(offpath2, 'w') as out:
        out2 = open(offpath2, 'w')
        for line in lines:
            out2.write(line + " 0\n")

def open_and_check(file_name, checkName):
    with open(file_name, "r",encoding="utf-8-sig") as f:
        json_data = json.load(f)
    for count in range(0,len(json_data["tooth"])):
        if(str(json_data["tooth"][count]["number"]) != checkName):
            continue
        else :
            return json_data["tooth"][count]["label"]

def Read_Lower_C(Lower_C):
    Lower_C_Dict = dict()
    with open(Lower_C, 'r') as f:
        lines = f.readlines()
    index = 0
    for line in lines:
        if line.startswith('#'):
            input_line = line.rstrip()
            Lower_C_Dict[input_line] = index
        index += 1
    return lines, Lower_C_Dict

def Read_Upper_C(Upper_C):
    Upper_C_Dict = dict()
    with open(Upper_C, 'r') as f:
        lines = f.readlines()
    index = 0
    for line in lines:
        if line.startswith('#'):
            input_line = line.rstrip()
            Upper_C_Dict[input_line] = index
        index += 1
    return lines, Upper_C_Dict

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
    if os.path.exists(temppath) == False:
        os.mkdir(temppath)
    for file in file_list:
        if file[-9:-8] == 'U':
            output_dir = resultpath + '/upper/' + file[:-8]
            fufu(returnpath + r"/" + file, output_dir)
        elif file[-9:-8] == 'L':
           output_dir = resultpath + '/lower/' + file[:-8]
           fufu(returnpath + r"/" + file, output_dir)

def obj_to_vtp():
    for i in os.listdir('result/lower'):
        if os.path.isfile(f'./result_vtp/lower/{i}.vtp'):
            continue
        elif i.startswith('.'):
            continue
        else:
        # load obj file
            data  = []
            mesh = load('result/lower/' + i + '/Sample_0.obj')
            obj = open('result/lower/' + i + '/Sample_0.obj','r')
            lines = obj.readlines()
            for line in lines:
                if line.startswith('g'):
                    data.append(round(float(line[9:-1]),1))    
            # get the cell array namely 'GroupIds'
            label_array = mesh.celldata['GroupIds']
            for label in range(len(label_array)):
                if label_array[label] == 0.0:
                    continue
                else: 
                    index = int(label_array[label])
                    label_array[label] = data[index]
            # assign this cell array to a new cell array called 'Label', which is used in Mesh Labeler
            mesh.celldata['Label'] = label_array # note this one is not working in latest vedo; my version is 2021.0.3]
            # delete un-used array (option)
            mesh.polydata().GetCellData().RemoveArray('GroupIds')
            mesh.polydata().GetPointData().RemoveArray('Normals')
            # write the mesh object to VTP format
            write(mesh, f'./result_vtp/lower/{i}.vtp')
    for i in os.listdir('result/upper'):
        if os.path.isfile(f'./result_vtp/upper/{i}.vtp'):
            continue
        elif i.startswith('.'):
            continue
        else:
            data  = []
            mesh = load('result/upper/' + i + '/Sample_0.obj')
            obj = open('result/upper/' + i + '/Sample_0.obj','r')
            lines = obj.readlines()
            for line in lines:
                if line.startswith('g'):
                    data.append(round(float(line[9:-1]),1))    
            # get the cell array namely 'GroupIds'
            label_array = mesh.celldata['GroupIds']
            for label in range(len(label_array)):
                if label_array[label] == 0.0:
                    continue
                else: 
                    index = int(label_array[label])
                    label_array[label] = data[index]
            # load obj file
            mesh = load('result/upper/' + i + '/Sample_0.obj')
            # get the cell array namely 'GroupIds'
            label_array = mesh.celldata['GroupIds']
            # assign this cell array to a new cell array called 'Label', which is used in Mesh Labeler
            mesh.celldata['Label'] = label_array # note this one is not working in latest vedo; my version is 2021.0.3
            # delete un-used array (option)
            mesh.polydata().GetCellData().RemoveArray('GroupIds')
            mesh.polydata().GetPointData().RemoveArray('Normals')
            # write the mesh object to VTP format
            write(mesh, f'./result_vtp/upper/{i}.vtp')


def main():
    except_count = 0
    good_count = 0
    init_dirs(datapath)
    tooth_dict = {18:1, 17:2, 16:3, 15:4, 14:5, 13:6, 12:7, 11:8, 21:9, 22:10, 23:11, 24:12, 25:13, 26:14, 27:15, 18:16,
                48:1, 47:2, 46:3, 45:4, 44:5, 43:6, 42:7, 41:8, 31:9, 32:10, 33:11, 34:12, 35:13, 36:14, 37:15, 38:16}
    index = 0
    falseflag = False
    for original_file in original_file_list: 
        file_list = os.listdir(original_file)
        jsonFile = ""
        print(original_file)
        Lower_C = ""
        Upper_C = ""
        Lower_C_Copy = ""
        Upper_C_Copy = ""
        upper_label_count = 1
        lower_label_count = 1
        if os.path.isdir(original_file + "\\" + file_list[0]) == True :
            index += 1
            continue
        
        for file in file_list:
            name = file[len(file)-6:]
            if name == 'C.json':
                jsonFile = original_file + "\\" + file
            elif file[len(file)-9:] == 'L-1-C.obj':
                Lower_C = original_file + "\\" + file
                Lower_C_Copy = returnpath + "\\" + file
                Full_Lower_C, Dict_Lower_C = Read_Lower_C(Lower_C)
            elif file[len(file)-9:] == 'U-1-C.obj':
                Upper_C = original_file + "\\" + file
                Upper_C_Copy = returnpath + "\\" + file
                Full_Upper_C, Dict_Upper_C = Read_Upper_C(Upper_C)
        
        if jsonFile == "":
            except_count += 2
            continue
        for file in file_list:
            objName = file[len(file)-4:]
            checkName = file[len(file)-6:]
            checkName = checkName[0:2]
            if objName == '.obj':
                if checkName == '-1' or checkName == '-C' or checkName == '_U' or checkName == '_L':
                    continue
                else:
                    label = open_and_check(jsonFile, checkName)
                    checkName = int(checkName)
                    if checkName > 10 and checkName < 30: # upper
                        upper_label_count += 1
                        check_label = tooth_dict.get(checkName)
                        check_name = '# mm_gid ' + str(label)
                        check_index = Dict_Upper_C.get(check_name)
                        if check_index == None:
                            print(file)
                            falseflag = True
                            continue
                        Full_Upper_C[check_index] = '# mm_gid ' + str(check_label) +'\n'
                        Full_Upper_C[check_index+1] = 'g mmGroup' + str(check_label) + '\n'
                    elif checkName > 30 and checkName < 50: # lower
                        lower_label_count += 1
                        check_label = tooth_dict.get(checkName)
                        check_name = '# mm_gid ' + str(label)
                        check_index = Dict_Lower_C.get(check_name)
                        if check_index == None:
                            print(file)
                            falseflag = True
                            continue
                        Full_Lower_C[check_index] = '# mm_gid ' + str(check_label) +'\n'
                        Full_Lower_C[check_index+1] = 'g mmGroup' + str(check_label) + '\n'
        if falseflag == True:
            falseflag = False
            continue
        if Lower_C_Copy == "":
            continue
        if Upper_C_Copy == "":
            continue
        
        upper_label_count_gt = 0
        for line in Full_Upper_C:
            if line.startswith('#'):
                upper_label_count_gt += 1
        
        if upper_label_count_gt != upper_label_count:
            except_count += 1
        else:
            if os.path.isfile(Upper_C_Copy) == True :
                good_count += 1
            else:
                with open(Upper_C_Copy, 'w') as out:
                    for line in Full_Upper_C:
                        out.write(line)
                    good_count += 1            

        lower_label_count_gt = 0
        for line in Full_Lower_C:
            if line.startswith('#'):
                lower_label_count_gt += 1
        
        if lower_label_count_gt != lower_label_count:
            except_count += 1
        else:
            if os.path.isfile(Lower_C_Copy) == True :
                good_count += 1
            else:
                with open(Lower_C_Copy, 'w') as out:
                    for line in Full_Lower_C:
                        out.write(line)
                    good_count += 1
        index += 1
    print('failed : '+ str(except_count))
    print('successed : ' + str(good_count))
    func(returnpath)
    obj_to_vtp()


if __name__ == "__main__":
	main()
