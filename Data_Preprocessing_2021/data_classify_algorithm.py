import os
import re
import json
path = os.path.dirname(os.path.abspath(__file__))
rootpath = path + '\\datas\\Labels_and_offs'
labelpath = rootpath + '\\labes'
datapath = path + '\\datas\\segmentation-files' # 여기로 자료들 input
import time
start = time.time()

original_file_list = list() #원래 파일 리스트
original_file_name_list = list() # 파일 이름만 저장된 리스트
copy_file_list = list() #복사된 파일 경로 리스트
copy_file_check = list() # 파일 경로 복사할 때 체크하기


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
        Pop = copy_file_check.pop()

def make_dirs():
    for lists in copy_file_list:
        os.makedirs(lists, exist_ok=True)

    for ten in range(1,5):
        for one in range(1,9):
            os.makedirs(rootpath+"\\labes\\"+str(ten*10+one), exist_ok=True)
            os.makedirs(rootpath+"\\labes\\"+str(ten*10+one)+"\\train", exist_ok=True)
            os.makedirs(rootpath+"\\labes\\"+str(ten*10+one)+"\\test", exist_ok=True)

def obj2off(objpath, offpath, offpath2):
    '''
    Convert obj file to off file
         :param objpath: path to the .obj file
         :param offpath: the save address of the path of the .off file
         :return: none
    '''
    line = ""

    vset = []
    fset = []
    with open(objpath,'r') as f:
        lines = f.readlines()
    p = re.compile(r'/+')
    space = re.compile(r' +')

    for line in lines:
                 #Get a line in the obj file as a string
        tailMark = " "
        line = line+tailMark
        if line[0]!='v' and line[0]!='f' :
            continue

        parameters = space.split(line.strip())
        if parameters[0] == "v": #if it is a vertex
            Point = []
            Point.append(eval( parameters[1]) )
            Point.append(eval( parameters[2]) )
            Point.append(eval( parameters[3]) )
            vset.append(Point)

        elif parameters[0] == "f": #if it is a face, the index of the vertices is stored
            vIndexSets = [] # collection of temporary storage points
            for i in range(1,len(parameters) ):
                x = parameters[i]
                ans = p.split(x)[0]
                index = eval(ans)
                index -= 1 #Because the vertex index starts at 1 in the obj file, and the vertex we store starts at 0, so we want to subtract 1
                vIndexSets.append(index)

            fset.append(vIndexSets)
    with open(offpath, 'w') as out:
        out = open(offpath, 'w')
        out.write("OFF\n")
        out.write(str(vset.__len__()) + " " + str(fset.__len__()) + " 0\n")
        for j in range(len(vset)):
            out.write(str(vset[j][0]) + " " + str(vset[j][1]) + " " + str(vset[j][2]) + "\n")

        for i in range(len(fset)):
            s = str(len( fset[i] ))
            for j in range( len( fset[i] ) ):
                s = s+ " "+ str(fset[i][j])
            s += "\n"
            out.write(s)
    with open(offpath2, 'w') as out:
        out2 = open(offpath2, 'w')
        out2.write("OFF\n")
        out2.write(str(vset.__len__()) + " " + str(fset.__len__()) + " 0\n")
        for j in range(len(vset)):
            out2.write(str(vset[j][0]) + " " + str(vset[j][1]) + " " + str(vset[j][2]) + "\n")

        for i in range(len(fset)):
            s = str(len( fset[i] ))
            for j in range( len( fset[i] ) ):
                s = s+ " "+ str(fset[i][j])
            s += "\n"
            out2.write(s)

    #print("{} converts to {} success!".format( p.split(objpath)[-1], p.split(offpath)[-1] ))

def open_and_check(file_name, checkName):
    #print(file_name)
    with open(file_name, "r",encoding="utf-8-sig") as f:
        json_data = json.load(f)

    #print(json.dumps(json_data) )
    for count in range(0,len(json_data["tooth"])):
        if(str(json_data["tooth"][count]["number"]) != checkName):
            continue
        else :
            for treatCount in range(0, len(json_data["tooth"][count]["treat"])):
                if(json_data["tooth"][count]["treat"][treatCount] == "Prep"):
                    return False
                if(json_data["tooth"][count]["treat"][treatCount] == "Implant"):
                    return False
                if(json_data["tooth"][count]["treat"][treatCount] == "RPD"):
                    return False
            for diagCount in range(0, len(json_data["tooth"][count]["diag"])):
                if(json_data["tooth"][count]["diag"] == ["Abrasion"]):
                    return False
                if(json_data["tooth"][count]["diag"] == ["Fracture"]):
                    return False
                if(json_data["tooth"][count]["diag"] == ["Loss-RR"]):
                    return False

            return True



def main():
    init_dirs(datapath) #우선 기본적인 저장하기.
    make_dirs() #파일 만들기

    ###################################
    
    index = 0
    for original_file in original_file_list: #original_file은 파일들의 이름.
        file_list = os.listdir(original_file) #여기에서 읽어온다.
        jsonFile = ""
        #print(original_file)

        if os.path.isdir(original_file + "\\" + file_list[0]) == True :
            index += 1
            continue
        
        for file in file_list:
            name = file[len(file)-5:]
            if name == '.json':
                jsonFile = original_file + "\\" + file
                break

        for file in file_list:
            objName = file[len(file)-4:]
            checkName = file[len(file)-6:]
            checkName = checkName[0:2]
            if objName == '.obj': #obj 파일 중에서도
                if checkName == '-1' or checkName == '-C' or checkName == '_U' or checkName == '_L': #뒤에 라벨이 붙은 것 아니면 다 continue
                    continue
                else:
                    if open_and_check(jsonFile, checkName): #open_and_check 통과하면, 저장하면 됨
                        copy_file = file[:-4]
                        copy_file += '.off'
                        obj2off(original_file + "\\" + file,copy_file_list[index] + "\\" + copy_file, labelpath+ "\\" + checkName + "\\"+copy_file)
        print(original_file + "  finish!")
        index += 1



if __name__ == "__main__":
	main()
