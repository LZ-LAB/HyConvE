# -- coding: utf-8 --
# @Time: 2022-04-05 11:08
# @Author: WangCx
# @File: process
# @Project: HypergraphNN_test

ary_2 = []
ary_3 = []
ary_4 = []
ary_5 = []
ary_6 = []

with open("./test.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        rel = line[0]
        ents = line[1:]
        if len(ents) == 2:
            ary_2.append([rel] + ents)
        elif len(ents) == 3:
            ary_3.append([rel] + ents)
        elif len(ents) == 4:
            ary_4.append([rel] + ents)
        elif len(ents) == 5:
            ary_5.append([rel] + ents)
        elif len(ents) == 6:
            ary_6.append([rel] + ents)
    f.close()




for i in [2, 3, 4, 5, 6]:
    with open("./test_{}.txt".format(i), "w") as f:
        for tuple_ in eval("ary_{}".format(i)):
            for k in range(len(tuple_)):
                if k != len(tuple_) - 1:
                    f.write(tuple_[k])
                    f.write("\t")
                else:
                    f.write(tuple_[k])
            f.write("\n")
        f.close()
