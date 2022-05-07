import os

dir_name = "blenderdata"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".mtl"):
        os.remove(os.path.join(dir_name, item))