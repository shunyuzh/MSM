from __future__ import print_function
from collections import Counter
import json
from tqdm import tqdm
import os
import jsonlines
from random import shuffle
global str
import argparse

# convert MS MARCO json file into the same format of biencoder-nq-train.json
def convert_nqstyle(input_file, out_file):
    all_list = []
    with open(input_file, "r+", encoding="utf8") as f:
        for item in tqdm(jsonlines.Reader(f)):
            item["question"] = item.pop("query")
            item["positive_ctxs"] = item.pop("positive_passages")
            item["hard_negative_ctxs"] = item.pop("negative_passages")
            item["negative_ctxs"] = item["hard_negative_ctxs"][:1]
            all_list.append(item)
    print("data size is :{}".format(len(all_list)))
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_list, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--input_file", default=None, type=str, required=True, help="Path the input file."
    )
    parser.add_argument(
        "--out_file", default=None, type=str, required=True, help="Path to the output file."
    )
    args = parser.parse_args()

    convert_nqstyle(args.input_file, args.out_file)