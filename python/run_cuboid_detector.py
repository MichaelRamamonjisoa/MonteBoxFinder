import argparse
import json
import os
from os import path as osp
import sys
import shutil


from multiprocessing import Pool

sys.path.append(".")

def parse_args():
    parser = argparse.ArgumentParser("Run Scene parsing with bounding boxes")
    parser.add_argument("--config", type=str,default="default_ransac", help="Config file to load from the configs folder")
    parser.add_argument("--scans_dir", type=str, default="", help="Path to Scannet scans")
    parser.add_argument("--scene_id", type=str, default=None, help="Optional Scene id, default will run on all scenes")
    parser.add_argument("--out_dir", type=str, default="", help="Path to store RANSAC results")
    parser.add_argument("--lib_dir", type=str, default="", help="Path to RANSAC code")
    parser.add_argument("--num_workers", type=int, default=1, help="number of threads")
    args = parser.parse_args()
    return args


class Runner:
    def __init__(self, args):
        self.scans_dir = args.scans_dir
        self.todo_scans = []
        self.lib_dir = args.lib_dir
        self.out_dir= args.out_dir
        self.num_workers = args.num_workers

        with open(osp.join("../configs", f"{args.config}.json"), "r") as f:
            self.ransac_config = json.load(f)

        if args.scene_id:
            self.todo_scans.append(args.scene_id)
        else:
            self.todo_scans = [scene.rsplit("scene")[-1] for scene in os.listdir(osp.join(self.scans_dir, "scans")) if
                               scene.startswith("scene")]
            self.todo_scans += [scene.rsplit("scene")[-1] for scene in
                                os.listdir(osp.join(self.scans_dir, "scans_test")) if scene.startswith("scene")]
            tmp = [scene_id for scene_id in self.todo_scans if self.is_scene_bad(scene_id)]
            num_total = len(self.todo_scans)
            self.todo_scans = tmp
            print(f"Found {num_total - len(self.todo_scans)} already done scans, {len(self.todo_scans)} left to do")
            print(f"You are about to run box detector on ALL scenes ({len(self.todo_scans)} in total)")
            print('Press any key to continue, or CTRL-C to exit.')
            key = input('')


    def run_on_scene(self, scene_id):

        #### RANSAC cuboid detection ####
        command = osp.join(self.lib_dir, "ransac_plane_detector")
        scene_path = osp.join(self.scans_dir, "scans", "scene" + scene_id, "scene" + scene_id + "_vh_clean_2.ply")
        if not os.path.exists(scene_path):
            scene_path = osp.join(self.scans_dir, "scans_test", "scene" + scene_id, "scene" + scene_id + "_vh_clean_2.ply")
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"The scan file {scene_path} does not exist")

        command += f" -m {scene_path} "
        options = f" -t {self.ransac_config['distance_threshold']}"
        options += f" -n {self.ransac_config['normals_threshold']}"
        options += f" -c {self.ransac_config['num_NN_normals']}"
        options += f" -k {self.ransac_config['min_cluster_size']}"
        options += f" -e {self.ransac_config['max_cluster_dist']}"
        command += options
        out_dir = osp.join(self.out_dir, f"results_scene{scene_id}", "clean")
        command += f" -o {out_dir}"
        if osp.exists(osp.join(out_dir, "pcloud_test.ply")):
            os.unlink(osp.join(out_dir, "pcloud_test.ply"))

        if osp.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        os.system(command)

        ### create symlink from scans to outdir ###
        command = f"ln -s {scene_path} {out_dir}/pcloud_test.ply"
        os.system(command)

    def is_scene_bad(self, scene_id):
        try:
            scene_path = osp.join(self.out_dir, f"results_scene{scene_id}", "clean")
            json_path = osp.join(scene_path, "pair_bboxes.json")
            bboxes = json.load(open(json_path, "r"))["bbox"]
            json_path = osp.join(scene_path, "single_bboxes.json")
            bboxes.append(json.load(open(json_path, "r"))["bbox"])
            return False
        except:
            return True


    def run(self):
        print(f"About to run on {len(self.todo_scans)}, do you confirm?")
        print('Press any key to continue, or CTRL-C to exit.')
        key = input('')
        with Pool(self.num_workers) as p:
            p.map(self.run_on_scene, self.todo_scans)

def main():
    args = parse_args()
    runner = Runner(args)
    runner.run()

if __name__ == "__main__":
    main()
