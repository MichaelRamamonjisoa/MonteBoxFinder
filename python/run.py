import argparse
from os import path as osp

import sys

sys.path.append(".")

from lib.observer import Observer
from lib.optimizer import *

import random
import numpy as np
import torch
# from multiprocessing import Pool
# from torch.multiprocessing import Pool, set_start_method


OPTIMIZERS = {
    "UH": UphillOptimizer,
    "UCB": UCBOptimizer,
}

def is_scene_done(results_dir, scene_id):
    try:        
        path = osp.join(results_dir, scene_id, "UH.json")
        data = json.load(open(path, "r"))
        assert len(data["bbox"]) > 0
        assert data["N_evals"] > 0
        path = osp.join(results_dir, scene_id, "UCB.json")
        data = json.load(open(path, "r"))
        assert len(data["bbox"]) > 0
        assert data["N_evals"] > 0
    except:
        return False
    return True


class Runner:
    def __init__(self, args):
        with open(osp.join("../configs", f"{args.config}.json"), "r") as f:
            self.cfg = json.load(f)
        self.is_benchmark = args.benchmark
        self.scans_dir = args.scans_dir
        self.outdir = args.outdir
        if torch.cuda.is_available() and not args.use_cpu:
            self.cfg["use_gpu"] = True
        else:
            self.cfg["use_gpu"] = False
        self.num_workers = args.num_workers
        self.observer_pool = {}
        self.show_results = args.show_results
        self.iterations_dict = None
        if args.num_iters_file:
            with open(args.num_iters_file, "r") as f:
                self.iterations_dict = json.load(f)

    def run_on_scene(self, scene_id):
        """
        run optimizer(s) on the scene
        """
        print(f"Running scene {scene_id}")
        current_cfg = copy.deepcopy(self.cfg)
        current_cfg["scene_rootdir"] = osp.join(self.scans_dir, f"results_scene{scene_id}", "clean")

        obs_pid = os.getpid()
        self.observer_pool[obs_pid] = Observer(current_cfg)
        self.observer_pool[obs_pid].sample_cloud()
        self.observer_pool[obs_pid].send_pcd_to_device()
        self.device = self.observer_pool[obs_pid].device

        if self.is_benchmark:
            optims_results = {}
            if not osp.exists(osp.join(self.outdir, scene_id)):
                os.makedirs(osp.join(self.outdir, scene_id))

            for method in OPTIMIZERS:
                optims_results[method] = {"score": 0, "bbox":[], "N_evals": 0}

            if self.iterations_dict is None:
                # run Uphill to get number of evaluations
                optimizer = OPTIMIZERS["UH"](current_cfg, self.observer_pool[obs_pid], self.device)
                solution, score, N_eval_UH = optimizer.run(show=False)
                optims_results["UH"]["score_history"] = optimizer.score_history
                optims_results["UH"]["score"] = float(score.cpu().numpy()) if isinstance(score, torch.Tensor) else score
                optims_results["UH"]["N_evals"] = N_eval_UH
                solution_ = [box.box_points.tolist() for box in solution]
                optims_results["UH"]["bbox"] = solution_
                savepath = osp.join(self.outdir, scene_id, "UH.json")
                with open(savepath, "w") as f:
                    json.dump(optims_results["UH"], f, indent=4)
            else:
                N_eval_UH = self.iterations_dict[scene_id]

            print(f"Running UCB with {N_eval_UH} evaluations")
            current_cfg["MAX_EVAL"] = N_eval_UH
            optimizer = OPTIMIZERS["UCB"](current_cfg, self.observer_pool[obs_pid], self.device)
            solution, score = optimizer.run(show=False)
            optims_results["UCB"]["score_history"] = optimizer.score_history
            optims_results["UCB"]["score"] = float(score.cpu().numpy()) if isinstance(score, torch.Tensor) else score
            optims_results["UCB"]["N_evals"] = N_eval_UH
            solution_ = [box.box_points.tolist() for box in solution]
            optims_results["UCB"]["bbox"] = solution_
            savepath = osp.join(self.outdir, scene_id, "UCB.json")
            with open(savepath, "w") as f:
                json.dump(optims_results["UCB"], f, indent=4)
        else:
            scene_out_path = osp.join(self.outdir, self.cfg["optimizer"], scene_id)
            if not osp.exists(scene_out_path):
                os.makedirs(scene_out_path)

            optimizer = OPTIMIZERS[self.cfg["optimizer"]](self.cfg, self.observer_pool[obs_pid], self.device)
            print(f"Running {self.cfg['optimizer']}, with {len(self.observer_pool[obs_pid].proposals)} boxes")
            if self.cfg["optimizer"] != "UH":
                S, scores = optimizer.run(show=self.show_results, savepath=scene_out_path)
                N_eval = optimizer.curr_iter
            else:
                S, scores, N_eval = optimizer.run(show=self.show_results, savepath=scene_out_path)

            scores = float(scores.cpu().numpy()) if isinstance(scores, torch.Tensor) else float(scores)
            print(f"Score: {scores} ({len(S)} boxes used, {N_eval} evaluations) ")
            optimizer.observer.show_selection(S)

        self.observer_pool.pop(obs_pid, None)

    def run(self, todo_scans):
        print(f"About to run on {len(todo_scans)}, do you confirm?")
        print('Press any key to continue, or CTRL-C to exit.')
        input('')
        with Pool(self.num_workers) as p:
            p.map(self.run_on_scene, todo_scans)


def parse_args():
    parser = argparse.ArgumentParser("Run Scene parsing with bounding boxes")
    parser.add_argument("--config", type=str,default="default", help="Config file to load from the configs folder")
    parser.add_argument("--scans_dir", type=str, default="", help="Path to RANSAC results")
    parser.add_argument("--scene_id", type=str, default=None, help="Scannet scene id")
    parser.add_argument("--use_cpu", action="store_true", help="Turn off usage of GPU")
    parser.add_argument("--outdir", default=".", type=str)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1, help="Currently, only single thread works")
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=0, type=int)
    parser.add_argument("--scene_list_file", default=None, type=str)
    parser.add_argument("--num_iters_file", default=None, type=str)
    parser.add_argument("--show", dest="show_results", action="store_true")
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    runner = Runner(args)
    if args.scene_id is not None and args.scene_id != "random":
        print(f"Running on scene {args.scene_id}")
        runner.run_on_scene(args.scene_id)
    elif args.scene_id == "random":
        todo_scenes = sorted([scene.rsplit("results_scene")[-1] for scene in os.listdir(args.scans_dir) if
                              scene.startswith("results_scene")])
        scene_id = random.choice(todo_scenes)
        runner.run_on_scene(scene_id)
    elif args.benchmark and args.scene_id is None:
        if args.scene_list_file is not None:
            with open(args.scene_list_file, "r") as f:
                todo_scenes = [x.strip() for x in f.readlines()]
            if todo_scenes[0].startswith("scene"):
                todo_scenes = [scene[5:] for scene in todo_scenes]
            todo_scenes = sorted(todo_scenes)
            tmp = sorted([scene_id for scene_id in todo_scenes if not is_scene_done(args.outdir, scene_id)])
            todo_scenes = tmp

            if args.end_idx != 0:
                todo_scenes = todo_scenes[args.start_idx:args.end_idx]
            else:
                todo_scenes = todo_scenes[args.start_idx:]

            print(f"Found  {len(todo_scenes)} left to do: continue?")
            print("Press any key to continue or CTRL + C to cancel")
            input('')
        else:
            if not args.force:
                print("Running benchmark on all scenes: continue?")
                print("Press any key to continue or CTRL + C to cancel")
                input('')
            todo_scenes = sorted([scene.rsplit("results_scene")[-1] for scene in os.listdir(args.scans_dir) if scene.startswith("results_scene")])
            num_total = len(todo_scenes)
            tmp = sorted([scene_id for scene_id in todo_scenes if not is_scene_done(args.outdir, scene_id)])
            todo_scenes = tmp

            if args.end_idx != 0:
                todo_scenes = todo_scenes[args.start_idx:args.end_idx]
            else:
                todo_scenes = todo_scenes[args.start_idx:]

            if not args.force:
                print(f"Found {num_total - len(todo_scenes)} done, {len(todo_scenes)} left to do: continue?")
                print("Press any key to continue or CTRL + C to cancel")
                input('')

        #TODO: find a way to make multiprocessing work
        for scene_id in todo_scenes:
            runner.run_on_scene(scene_id)


if __name__ == "__main__":
    main()
