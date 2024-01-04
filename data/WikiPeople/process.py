# -- coding: utf-8 --
# @Time: 2023-02-04 15:14
# @Author: WangCx
# @File: process
# @Project: HyConv_64_cross

import math
import random


file_name = ["train", "valid", "test"]

for name in file_name:
    two_ary = []
    other = []
    with open("./{}.txt".format(name)) as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            if len(line) == 3:
                two_ary.append(line)
            else:
                other.append(line)
        f.close()

    random.shuffle(two_ary)

    two_ary_5 = two_ary[:math.ceil(len(two_ary) * 0.05)] + other
    two_ary_10 = two_ary[:math.ceil(len(two_ary) * 0.1)] + other
    two_ary_20 = two_ary[:math.ceil(len(two_ary) * 0.2)] + other
    two_ary_40 = two_ary[:math.ceil(len(two_ary) * 0.4)] + other
    two_ary_50 = two_ary[:math.ceil(len(two_ary) * 0.5)] + other
    two_ary_60 = two_ary[:math.ceil(len(two_ary) * 0.6)] + other
    two_ary_80 = two_ary[:math.ceil(len(two_ary) * 0.8)] + other
    two_ary_90 = two_ary[:math.ceil(len(two_ary) * 0.9)] + other

    with open("./{}_5.txt".format(name), "w") as f:
        for tuple in two_ary_5:
            for i in range(len(tuple)):
                if i == len(tuple) - 1:
                    f.write(tuple[i])
                else:
                    f.write(tuple[i] + "\t")
            f.write("\n")
        f.close()

    with open("./{}_10.txt".format(name), "w") as f:
        for tuple in two_ary_10:
            for i in range(len(tuple)):
                if i == len(tuple) - 1:
                    f.write(tuple[i])
                else:
                    f.write(tuple[i] + "\t")
            f.write("\n")
        f.close()

    with open("./{}_20.txt".format(name), "w") as f:
        for tuple in two_ary_20:
            for i in range(len(tuple)):
                if i == len(tuple) - 1:
                    f.write(tuple[i])
                else:
                    f.write(tuple[i] + "\t")
            f.write("\n")
        f.close()

    with open("./{}_40.txt".format(name), "w") as f:
        for tuple in two_ary_40:
            for i in range(len(tuple)):
                if i == len(tuple) - 1:
                    f.write(tuple[i])
                else:
                    f.write(tuple[i] + "\t")
            f.write("\n")
        f.close()

    with open("./{}_50.txt".format(name), "w") as f:
        for tuple in two_ary_50:
            for i in range(len(tuple)):
                if i == len(tuple) - 1:
                    f.write(tuple[i])
                else:
                    f.write(tuple[i] + "\t")
            f.write("\n")
        f.close()

    with open("./{}_60.txt".format(name), "w") as f:
        for tuple in two_ary_60:
            for i in range(len(tuple)):
                if i == len(tuple) - 1:
                    f.write(tuple[i])
                else:
                    f.write(tuple[i] + "\t")
            f.write("\n")

        f.close()

    with open("./{}_80.txt".format(name), "w") as f:
        for tuple in two_ary_80:
            for i in range(len(tuple)):
                if i == len(tuple) - 1:
                    f.write(tuple[i])
                else:
                    f.write(tuple[i] + "\t")
            f.write("\n")

        f.close()

    with open("./{}_90.txt".format(name), "w") as f:
        for tuple in two_ary_90:
            for i in range(len(tuple)):
                if i == len(tuple) - 1:
                    f.write(tuple[i])
                else:
                    f.write(tuple[i] + "\t")
            f.write("\n")
        f.close()
