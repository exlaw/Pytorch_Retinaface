import os
gt_dir = "ground_truth"
tgt_dir = "resnet_self_train"
dir_dict = {}

for i in range(1, 11):
    filename = os.path.join(gt_dir, 'FDDB-fold-{}.txt'.format('%02d' % i))
    f = open(filename, 'r')
    lines = f.readlines()
    for line in lines:
        dir_dict[line.strip().replace('/', '_')] = i
    f.close()

with open(tgt_dir + "/FDDB_dets.txt", 'r') as f:
    while True:
        name = f.readline()[:-1].replace('/', '_')
        # print(name)
        # print(dir_dict[name])
        dir_name = os.path.join(tgt_dir, str(dir_dict[name]))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        filename = os.path.join(dir_name, name + ".txt")
        fw = open(filename, "w")
        line = f.readline()
        # print(line.strip())
        facenum = int(float(line.strip()))
        fw.write(name + "\n")
        fw.write(str(facenum) + "\n")
        for j in range(facenum):
            line = f.readline()
            fw.write(line)
        fw.close()



