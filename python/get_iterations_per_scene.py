import json
import argparse
import os
from os import path as osp

parser = argparse.ArgumentParser()
parser.add_argument("--rootdir", type=str, default=".")
parser.add_argument("--outdir", type=str, default=".")

args = parser.parse_args()

outpath = osp.join(args.outdir, "num_iterations_fulldataset_noduplicates.json")

iters_dict = {}

for scene_id in os.listdir(args.rootdir):
    result_path = osp.join(args.rootdir, scene_id, "UH.json")
    with open(result_path, "r") as f:
        num_iters = json.load(f)["N_evals"]

    iters_dict[str(scene_id)] = num_iters

with open(outpath, "w") as f:
    json.dump(iters_dict, f, indent=4)

