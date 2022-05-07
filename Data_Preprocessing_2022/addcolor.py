import os

for i in os.listdir('originaldata/'):
    with open(f'originaldata/{i}', 'r') as f:
        lines = f.readlines()
        with open(f'colordata/{i}', 'w') as out_file:
            for line in lines:
                if line.startswith('vn'):
                    out_file.write(line)
                elif line.startswith('v'):
                        out_file.write(line.strip()+' 1.000000 1.000000 0.000000'+'\n')
                        # print(line + '0.0 0.0 0.0')
                else:
                    out_file.write(line)
            f.close()
            out_file.close()