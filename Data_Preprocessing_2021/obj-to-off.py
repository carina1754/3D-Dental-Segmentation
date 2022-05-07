import re
import os

def obj2off(objpath, offpath):
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

    print("{} converts to {} success!".format( p.split(objpath)[-1], p.split(offpath)[-1] ))
         
def main():
    # python 3에서는 print() 으로 사용합니다.
    obj2off('C:/example/Label sample 1/1_1F0021.obj','C:/example/Output sample/1_1F0021.off')

if __name__ == "__main__":
   main()
