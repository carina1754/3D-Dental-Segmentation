
# Using the with statement the file is automatically closed

with open("a.obj", 'r') as infile:
    data = infile.readlines()
with open("a.obj", 'w') as outfile:
    for i in data:
        if not i.startswith("vn"):
            if i.startswith("f"):
                face = i.split(' ')
                face_scale = ""
                for f in range(4):
                    if f == 0:
                        face_scale += "f "
                    else:        
                        face_length = int(len(face[f])/2-1)
                        scale = face[f]
                        face_part = scale[:face_length]
                        face_scale += face_part + " "
                face_scale += "\n"
                outfile.write(face_scale)
            elif i.startswith("v"):
                ver = i.split(' ')
                ver_scale = ""
                for v in range(4):
                    if v == 0:
                        ver_scale += "v "
                    else:        
                        ver_scale += ver[v] + " "
                ver_scale += "\n"
                outfile.write(ver_scale)
            else:
                outfile.write(i)