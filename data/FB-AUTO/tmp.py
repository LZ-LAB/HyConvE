# -- coding: utf-8 --
# @Time: 2022-03-27 19:35
# @Author: WangCx
# @File: tmp
# @Project: HypergraphNN


file = ["train.txt", "test.txt", "valid.txt"]
id2ent = {}
ent2id = {}
i = 0
for filename in file:
    with open("./{}".format(filename)) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            ents = line[1:]
            for ent in ents:
                if ent not in ent2id:
                    ent2id[ent] = i
                    i+=1

print(ent2id)
print(len(ent2id))