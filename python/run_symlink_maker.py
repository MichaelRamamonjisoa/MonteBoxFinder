import argparse
import os
from os import path as osp
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser("Run Scene parsing with bounding boxes")
    parser.add_argument("--config", type=str,default="default", help="Config file to load from the configs folder")
    parser.add_argument("--scans_dir", type=str, default="", help="Path to RANSAC results")
    parser.add_argument("--outdir", default=".", type=str)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    return args


class Runner:
    def __init__(self, args):
        self.scans_dir = args.scans_dir
        self.todo_scans = []
        self.out_dir= args.outdir
        self.num_workers = args.num_workers

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
        scene_path = osp.join(self.scans_dir, "scans", "scene" + scene_id, "scene" + scene_id + "_vh_clean_2.ply")
        if not os.path.exists(scene_path):
            scene_path = osp.join(self.scans_dir, "scans_test", "scene" + scene_id, "scene" + scene_id + "_vh_clean_2.ply")
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"The scan file {scene_path} does not exist")

        out_dir = osp.join(self.out_dir, f"results_scene{scene_id}", "clean")
        try:
            os.readlink(osp.join(out_dir, "pcloud_test.ply"))
            os.unlink(osp.join(out_dir, "pcloud_test.ply"))
            os.remove(osp.join(out_dir, "pcloud_test.ply"))
        except Exception as e:
            pass            

        ### create symlink from scans to outdir ###
        command = f"ln -s {os.path.abspath(scene_path)} {out_dir}/pcloud_test.ply"
        os.system(command)

    def is_scene_bad(self, scene_id):
        return True

    def run(self):
        print(f"About to run on {len(self.todo_scans)}, do you confirm?")
        print('Press any key to continue, or CTRL-C to exit.')
        key = input('')
        with Pool(self.num_workers) as p:
            p.map(self.run_on_scene, self.todo_scans)
        # for scene_id in self.todo_scans:
        #     self.run_on_scene(scene_id)


def main():
    args = parse_args()
    runner = Runner(args)
    runner.run()
if __name__ == "__main__":
    main()
