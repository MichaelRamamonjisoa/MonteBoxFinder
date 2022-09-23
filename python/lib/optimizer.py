import copy

import numpy as np
import torch
import time
import json

import os

from lib.scoring import Scoring
from lib.visualization_utils import AverageMeter

import random

class Optimizer:
    def __init__(self, cfg, observer, device=torch.device("cpu"), shuffle_boxes=True):
        self.cfg = cfg
        self.max_eval = self.cfg["MAX_EVAL"]
        self.tolerance = self.cfg["tolerance"]
        self.device = device

        # observer and scorer
        self.observer = observer
        if shuffle_boxes:
            forward_shuffle_indices = np.arange(len(self.observer.proposals))
            self.observer.proposals = [self.observer.proposals[idx] for idx in forward_shuffle_indices]  # those will be shuffled in the observer too
        self.scoring = Scoring(cfg, observer, device)
        self.best_loss = np.inf
        self.loss = 0
        self.best_solution = []
        self.curr_iter = 0

        # average meters for timings
        self.obs_sample_cloud_am = AverageMeter()
        self.score_sample_cloud_am = AverageMeter()
        self.simulate_am = AverageMeter()
        self.opt_update_am = AverageMeter()
        self.avg_pts_am = AverageMeter()

    def reset(self):
        self.best_loss = np.inf
        self.loss = 0
        self.best_solution = []
        # initialize UCB states
        self.best_loss_per_proposal = 10 * np.ones((len(self.observer.proposals), 2))
        self.all_n_1 = np.ones((len(self.observer.proposals)))
        self.all_n_0 = np.ones((len(self.observer.proposals)))
        self.all_priors = 0.5 * np.ones((len(self.observer.proposals)))
        self.all_UCB = np.zeros((len(self.observer.proposals), 2))

        for d in self.observer.proposals:
            d.n_0 = 1
            d.n_1 = 1
            d.best_loss_0 = 10
            d.best_loss_1 = 10

    def initialize_best_losses_and_priors(self):
        for p in self.observer.proposals:
            p.best_loss_1 = 10
            p.best_loss_0 = 10
            p.n_1 = 1
            p.n_0 = 1
            p.prior_1 = p.best_loss_0 / (p.best_loss_0 + p.best_loss_1)

    def best_solution_based_on_priors(self):
        simulation = []
        box_indices = np.argsort(self.all_priors)[::-1]
        self.observer.proposals = [self.observer.proposals[i] for i in box_indices.tolist()]

        for p in self.observer.proposals:
            if self.all_priors[p.idx] > 0.5:
                compatible_box = True
                for q in simulation:
                    if not self.observer.is_compatible[q.idx, p.idx]:
                        compatible_box = False
                        break
                if compatible_box:
                    simulation.append(p)
        loss = self.scoring.scoring_loss(simulation)
        return (simulation, loss)

    def best_solution_based_on_UCB(self):
        simulation = []
        for h in self.observer.proposals:
            if self.all_UCB[h.idx, 1] > self.all_UCB[h.idx, 0]:
                compatible_box = True
                for q in simulation:
                    if not self.observer.is_compatible[q.idx, h.idx]:
                        compatible_box = False
                        break
                if compatible_box:
                    simulation.append(h)
        loss = self.scoring.scoring_loss(simulation)
        return (simulation, loss)

    def save_solution(self, solution, score, path, color=[0, 1, 0]):
        """
        Save solution in a Json file
        :param solution : (List of Bbox) the list of bounding boxes
        :param path : (str) path to Json file
        :return:
        """
        out_points = [box.box_points.tolist() for box in solution]
        with open(path, "w") as f:
            json.dump({"bbox": out_points, "score": score, "color": color}, f)

    def run(self):
        raise NotImplementedError

    def simulate(self):
        raise NotImplementedError

    def compute_loss_and_update_state(self, simulation):
        raise NotImplementedError

##############################################################

class UphillOptimizer(Optimizer):
    def __init__(self, cfg, observer, device=torch.device("cpu"), shuffle_boxes=False):
        super(UphillOptimizer, self).__init__(cfg, observer, device, shuffle_boxes)

    def run(self, show=True, savepath=None):

        for h in self.observer.proposals:
            h.available = True

        solution = []
        best_loss = np.inf
        new_best_loss = np.inf
        loss = 0
        N_evaluations = 0
        ok = True
        tries = 0

        # resample a new sub pointcloud
        # generate self.observer.observed_cloud
        self.observer.sample_cloud()
        # generate self.observer.gt_gpu
        self.observer.send_pcd_to_device()
        self.score_history = {}

        while ok:
            tries += 1
            print(f"try {tries}")
            best_hypothesis = None

            for idx, h in enumerate(self.observer.proposals):
                if h.available:
                    indices = [d.idx for d in solution if d.idx != h.idx]
                    if not np.all(self.observer.is_compatible[indices, h.idx]):
                        h.available = False
                    else:
                        solution.append(h)
                        s_sample_cloud = time.time()
                        synth_cloud, synth_normals = self.scoring.sample_from_proposals(solution, self.device)
                        self.score_sample_cloud_am.update(time.time() - s_sample_cloud)

                        # compute loss
                        loss = self.scoring.scoring_loss(solution)

                        N_evaluations += 1
                        if N_evaluations % max(100, int(0.01 * N_evaluations / self.max_eval)) == 0:
                            print(f"evals {N_evaluations}/{self.max_eval}")
                        if loss < new_best_loss:
                            new_best_loss = loss
                            best_hypothesis = h
                            best_loss = float(new_best_loss.cpu().numpy()) if isinstance(new_best_loss,
                                                                                     torch.Tensor) else new_best_loss
                            self.score_history[str(N_evaluations)] = best_loss
                            if savepath is not None:
                                # save intermediate result
                                self.save_solution(solution, best_loss,
                                                   os.path.join(savepath, "iter_{:03d}.json".format(N_evaluations)))
                        solution = solution[:-1]

            if best_hypothesis != None:
                solution.append(best_hypothesis)
                best_loss = new_best_loss
                best_loss = float(best_loss.cpu().numpy()) if isinstance(best_loss, torch.Tensor) else best_loss
                self.score_history[str(N_evaluations)] = best_loss
                if show:
                    print(" --> %f" % best_loss)
                    self.observer.show_selection(copy.deepcopy(solution), thickness=self.cfg["plot_thickness"])
                best_hypothesis.available = False

                if savepath is not None:
                    # save intermediate result
                    self.save_solution(solution, best_loss,
                                       os.path.join(savepath, "iter_{:03d}.json".format(N_evaluations)))
            else:
                print("no best hypothesis")
                ok = False
                self.score_history[str(0)] = self.score_history[str(np.min([int(k) for k in self.score_history.keys()]))] # adds first best reached value as index 0
                self.score_history[str(N_evaluations)] = float(best_loss.cpu().numpy()) if isinstance(best_loss,
                                                                                                      torch.Tensor) else best_loss
            if self.max_eval > -1 and N_evaluations >= self.max_eval:
                ok = False
        if show:
            print("%d evaluations used" % N_evaluations)
            print("Avg number of points: {:.2f}".format(self.avg_pts_am.avg))

        #         solution = self.scoring.renderer.remove_invisible_objects(solution)

        return solution, best_loss, N_evaluations


class UCBOptimizer(Optimizer):
    def __init__(self, cfg, observer, device=torch.device("cpu"), shuffle_boxes=True):
        super(UCBOptimizer, self).__init__(cfg, observer, device, shuffle_boxes)
        self.initialize_best_losses_and_priors()
        # update all UCB scores
        self.delta = self.cfg["delta_UCB"]
        self.delta_final = self.cfg["delta_UCB_final"]
        self.epsilon = self.cfg["epsilon_UCB"]
        self.epsilon_final = self.cfg["epsilon_UCB_final"]
        self.reset()

    def reset(self):
        super(UCBOptimizer, self).reset()

        curr_iter = 1

        self.all_UCB[:, 0] = -self.best_loss_per_proposal[:, 0] + \
                             self.delta * np.sqrt(np.log(curr_iter) / self.all_n_0[:])
        self.all_UCB[:, 1] = -self.best_loss_per_proposal[:, 1] + \
                             self.delta * np.sqrt(np.log(curr_iter) / self.all_n_1[:])

    def simulate_random(self):
        simulated_scene = []
        for d in self.observer.proposals:
            if np.random.sample() < 0.5:
                compatible_box = True
                indices = [d_j.idx for d_j in simulated_scene if d_j.idx != d.idx]
                if not np.all(self.observer.is_compatible[d.idx, indices]):
                    compatible_box = False

                if compatible_box:
                    simulated_scene.append(d)
                    self.is_proposal_in_simulation[d.idx] = True
        return simulated_scene


    def simulate(self):
        simulated_scene = []

        curr_eps = (1. - self.curr_iter / self.max_eval) * self.epsilon + (self.curr_iter / self.max_eval) * self.epsilon_final
        for d in self.observer.proposals:
            if np.random.sample() >= curr_eps:
                if self.all_UCB[d.idx, 1] > self.all_UCB[d.idx, 0]:
                    compatible_box = True
                    indices = [d_j.idx for d_j in simulated_scene if d_j.idx != d.idx] 
                    if not np.all(self.observer.is_compatible[indices, d.idx]):
                        compatible_box = False
                    if compatible_box:
                        simulated_scene.append(d)
                        self.is_proposal_in_simulation[d.idx] = True
            else:
                if self.all_UCB[d.idx, 1] < self.all_UCB[d.idx, 0]:
                    compatible_box = True
                    indices = [d_j.idx for d_j in simulated_scene if d_j.idx != d.idx]
                    if not np.all(self.observer.is_compatible[indices, d.idx]):
                        compatible_box = False

                    if compatible_box:
                        simulated_scene.append(d)
                        self.is_proposal_in_simulation[d.idx] = True

        return simulated_scene

    def compute_loss_and_update_state(self, simulation):

        with torch.no_grad():
            self.loss = self.scoring.scoring_loss(simulation)
        loss_local = self.loss.cpu().numpy()

        condition_UCB_0 = np.logical_and(self.best_loss_per_proposal[:, 0] > loss_local,
                                          np.logical_not(self.is_proposal_in_simulation))
        self.all_n_0[~self.is_proposal_in_simulation] += 1
        self.best_loss_per_proposal[condition_UCB_0, 0] = loss_local

        condition_UCB_1 = np.logical_and(self.best_loss_per_proposal[:, 1] > loss_local,
                                          self.is_proposal_in_simulation)
        self.all_n_1[self.is_proposal_in_simulation] += 1
        self.best_loss_per_proposal[condition_UCB_1, 1] = loss_local

        self.all_priors = self.best_loss_per_proposal[:, 0] / \
                          (self.best_loss_per_proposal[:, 0] + self.best_loss_per_proposal[:, 1])

        curr_iter = self.curr_iter + 1
        curr_delta =  (1. - curr_iter / self.max_eval) * self.delta + (curr_iter / self.max_eval) * self.delta_final

        self.all_UCB[condition_UCB_0, 0] = -self.best_loss_per_proposal[condition_UCB_0, 0] + \
                                           curr_delta * np.sqrt(np.log(curr_iter) / self.all_n_0[condition_UCB_0])
        self.all_UCB[condition_UCB_1, 1] = -self.best_loss_per_proposal[condition_UCB_1, 1] + \
                                           curr_delta * np.sqrt(np.log(curr_iter) / self.all_n_1[condition_UCB_1])


    def run(self, show=True, savepath=None):
        if savepath is not None:
            if not os.path.exists(savepath):
                os.makedirs(savepath)

        N_eval_MC = self.cfg["MC_UCB"]
        self.is_proposal_in_simulation = np.zeros(len(self.observer.proposals)).astype("bool")
        self.score_history = {}

        # Initialize with random choice
        for i in range(N_eval_MC):
            self.is_proposal_in_simulation *= False
            simulation = self.simulate_random()

            if len(simulation) > 0:
                self.compute_loss_and_update_state(simulation)

                if self.loss < self.best_loss:
                    self.best_loss = self.loss
                    self.score_history[str(self.curr_iter + i)] = float(self.best_loss.cpu().numpy())
                    self.best_solution = copy.deepcopy(simulation)

                self.curr_iter += 1

        self.is_proposal_in_simulation = np.zeros(len(self.observer.proposals)).astype("bool")

        start_time = time.time()
        for i in range(self.max_eval - N_eval_MC):
            timing_period = int(self.max_eval / 20)
            if self.max_eval > 10 and i % timing_period == 1:
                print("iter {}/{} ({:.4f} s/it)".format(self.curr_iter, self.max_eval,
                                                        (time.time() - start_time) / timing_period))
                start_time = time.time()
            self.is_proposal_in_simulation *= False

            # sort by UCB score
            box_indices = np.argsort(self.all_UCB[:, 1])[::-1]
            self.observer.proposals = [self.observer.proposals[i] for i in box_indices.tolist()]

            s_simulate = time.time()
            simulation = self.simulate()
            self.simulate_am.update(time.time() - s_simulate)

            if len(simulation) > 0:
                s_sample_cloud = time.time()
                self.score_sample_cloud_am.update(time.time() - s_sample_cloud)

                s_update = time.time()
                self.compute_loss_and_update_state(simulation)
                self.opt_update_am.update(time.time() - s_update)

                if self.loss < self.best_loss:
                    if show:
                        print("i = %d, best loss = %f" % (i, self.loss))
                        self.observer.show_selection(copy.deepcopy(simulation), thickness=self.cfg["plot_thickness"])
                    print("i = %d, best loss (same) = %f" % (i, self.best_loss))

                    if savepath is not None:
                        # save intermediate result
                        self.save_solution(simulation, float(self.loss.cpu().numpy()),
                                           os.path.join(savepath, "iter_{:03d}.json".format(i)))

                    self.best_loss = self.loss
                    self.score_history[str(self.curr_iter + N_eval_MC)] = float(self.best_loss.cpu().numpy())
                    self.best_solution = copy.deepcopy(simulation)
                else:
                    if show and i % 100 == 1:
                        print("i = %d, best loss (same) = %f" % (i, self.best_loss))

            self.curr_iter += 1

        (best_sol2, best_loss2) = self.best_solution_based_on_priors()
        if best_loss2 < self.best_loss:
            if show:
                print("Using solution based on priors, loss = %f" % best_loss2)
            self.best_solution = best_sol2
            self.best_loss = best_loss2

            if savepath is not None:
                # save intermediate result
                self.save_solution(self.best_solution, float(self.loss.cpu().numpy()),
                                   os.path.join(savepath, "iter_{:03d}.json".format(i)))

        (best_sol2, best_loss2) = self.best_solution_based_on_UCB()
        if best_loss2 < self.best_loss:
            if show:
                print("Using solution based on UCB, loss = %f" % best_loss2)
            self.best_solution = best_sol2
            self.best_loss = best_loss2

            if savepath is not None:
                # save intermediate result
                self.save_solution(self.best_solution, float(self.loss.cpu().numpy()),
                                   os.path.join(savepath, "iter_{:03d}.json".format(i)))

        self.score_history[str(self.curr_iter + N_eval_MC)] = float(self.best_loss.cpu().numpy())

        print("===== Runtimes =====")
        print("Sample Observer: {:.2f}".format(self.obs_sample_cloud_am.avg))
        self.obs_sample_cloud_am.reset()
        print("Sample Scorer: {:.2f}".format(self.score_sample_cloud_am.avg))
        self.score_sample_cloud_am.reset()
        print("UCB simulate: {:.2f}".format(self.simulate_am.avg))
        self.simulate_am.reset()
        print("UCB update: {:.2f}".format(self.opt_update_am.avg))
        self.opt_update_am.reset()

        return self.best_solution, self.best_loss
