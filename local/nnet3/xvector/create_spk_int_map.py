#!/bin/local/env python
# Copyright 2022  Johns Hopkins University (Author: Desh Raj)

import sys

in_dir = sys.argv[1]

# Read speaker ids and get int mapping
spk2int = {}
with open(in_dir + "/spk2utt") as f:
    for line in f:
        spk = line.split()[0]
        if spk.startswith("multi"):
            spk1, spk2 = spk[6:].split("_")  # remove multi-
            if spk1 not in spk2int:
                spk2int[spk1] = len(spk2int)
            if spk2 not in spk2int:
                spk2int[spk2] = len(spk2int)
        else:
            spk = spk[7:]  # remove single-
            if spk not in spk2int:
                spk2int[spk] = len(spk2int)

# Write the mapping to a file
with open(in_dir + "/spk2int", "w") as f:
    for spk in spk2int:
        f.write("{0} {1}\n".format(spk, spk2int[spk]))

# Create utt2int
with open(in_dir + "/utt2spk") as f_in, open(in_dir + "/utt2int", "w") as f_out:
    for line in f_in:
        utt, spk = line.split()
        if spk.startswith("multi"):
            spk1, spk2 = spk[6:].split("_")
            spk1_int = spk2int[spk1]
            spk2_int = spk2int[spk2]
            f_out.write("{0} {1},{2}\n".format(utt, spk1_int, spk2_int))
        else:
            spk_int = spk2int[spk[7:]]
            f_out.write("{0} {1}\n".format(utt, spk_int))
