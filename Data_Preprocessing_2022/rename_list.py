import os
i_sample = 0
file_list = os.listdir("result_vtp\\lower")
for file in file_list:
    file_oldname = os.path.join("result_vtp\\lower", file)
    file_newname_newfile = os.path.join("rename_vtp\\lower", "Sample_0{0}_d.vtp".format(i_sample))
    i_sample += 1
    os.rename(file_oldname, file_newname_newfile)