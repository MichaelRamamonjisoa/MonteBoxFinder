import copy

import open3d as o3d
import torch
import numpy as np
from os import path as osp
import json

from lib.visualization_utils import bboxes_to_thickboxes
from lib.boxes import Bbox
from lib.box_utils import boxes_to_mesh_np
import time
from tqdm import tqdm
from ordered_set import OrderedSet


class Observer:
    """
    Class that handles the scene point cloud and detected bounding boxes
    self.full_pcd (o3d.geometry.PointCloud) contains the full point cloud:
        self.full_pcd.points (o3d.utility.Vector3dVector): 3D points
        self.full_pcd.normals (o3d.utility.Vector3dVector): 3D normals

    self.observed_cloud (o3d.geometry.PointCloud) contains the subset point cloud that acts as ground truth for loss computation
    self.gt_gpu (torch.Tensor) is a copy of self.observed_cloud in torch.Tensor format
    """
    def __init__(self, cfg):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() and cfg["use_gpu"] else torch.device("cpu")
        self.single_boxes = []
        self.pair_boxes = []
        self.get_single_boxes(cfg["scene_rootdir"])
        num_single_boxes = len(self.single_boxes)
        self.get_pair_boxes(cfg["scene_rootdir"])

        self.cfg = cfg
        self.tolerance = self.cfg["tolerance"]
        self.proposals = self.merge_boxes()
        self.num_proposals = len(self.proposals)
        self.precompute_compatibilities()
        self.get_single_boxes(cfg["scene_rootdir"])
        self.get_pair_boxes(cfg["scene_rootdir"])
        self.proposals = self.merge_boxes()
        self.num_proposals = len(self.proposals)


        self.num_points = self.cfg["num_scene_points"]
        self.observed_cloud = None
        self.full_pcd = None
        self.gt_gpu = None
        self.get_gt_pointcloud()


    def precompute_compatibilities(self):
        N = len(self.proposals)
        filepath = osp.join(self.cfg["scene_rootdir"], "compatibility_matrix.npy")
        try:
            self.is_compatible = np.load(filepath)
            assert self.is_compatible.ndim == 2 and self.is_compatible.shape[0] == N and self.is_compatible.shape[1] == N
        except:
            print("Compatibility matrix is either corrupted or not computed")
            self.is_compatible = torch.zeros((N, N), dtype=torch.bool).to(self.device)

            batch_size = 1500
            for i in tqdm(range(N-1)):
                d_query = self.proposals[i]
                # we batch the computation of IoUs to prevent GPU OOM
                if batch_size > len(self.proposals):
                    batch_size = len(self.proposals)
                for j in range(len(self.proposals) // batch_size):
                    # print(f"batch {j} / {len(self.proposals) // batch_size}")
                    start = j * batch_size
                    end = (j + 1) * batch_size
                    d_others = [d for d in self.proposals[i+1+start:i+1+end]]
                    if len(d_others) != 0:
                        self.is_compatible[i, i+1+start:i+1+end], true_ious = d_query.compatible(d_others[:end-start],
                                                                                                 self.tolerance)
                        self.is_compatible[i+1+start:i+1+end, i] = self.is_compatible[i, i+1+start:i+1+end].T

            self.is_compatible = self.is_compatible.cpu().numpy()

            np.save(filepath, self.is_compatible)

        for proposal in self.proposals:
            compatibility_row = self.is_compatible[proposal.idx]
            compatible_boxes_idx = np.where(compatibility_row)[0]
            incompatible_boxes_idx = np.where(~compatibility_row)[0]

            proposal.compatible_boxes |= compatible_boxes_idx.tolist()
            proposal.incompatible_boxes |= incompatible_boxes_idx.tolist()

        self.num_proposals = len(self.proposals)

        # update incompatible_boxes OrderedSet for each proposal
        for new_id, proposal in enumerate(self.proposals):
            compatibility_row = self.is_compatible[new_id]
            compatible_boxes_idx = np.where(compatibility_row)[0]
            incompatible_boxes_idx = np.where(~compatibility_row)[0]

            proposal.compatible_boxes = OrderedSet(compatible_boxes_idx.tolist())
            proposal.incompatible_boxes = OrderedSet(incompatible_boxes_idx.tolist())

    def get_gt_pointcloud(self):
        gt_pc_path = osp.join(self.cfg["scene_rootdir"], "pcloud_test.ply")

        tmp = o3d.io.read_triangle_mesh(gt_pc_path)
        self.full_pcd = o3d.geometry.PointCloud()
        self.full_pcd.points = tmp.vertices
        if tmp.has_vertex_colors():
            colors = np.array(tmp.vertex_colors, dtype="float")
            if np.max(colors) > 1:
                colors /= 255
        else:
            colors = np.zeros((len(tmp.vertices), 3))

        self.full_pcd.colors = o3d.utility.Vector3dVector(colors)
        if tmp.has_triangles():
            tmp = tmp.compute_triangle_normals()
            tmp = tmp.compute_vertex_normals()
            self.full_pcd.normals = tmp.vertex_normals
        else:
            self.full_pcd.estimate_normals()
        del tmp

    def sample_cloud(self):
        r = np.arange(len(self.full_pcd.points))
        np.random.shuffle(r)
        self.observed_cloud = o3d.geometry.PointCloud()
        self.observed_cloud.points = o3d.utility.Vector3dVector(np.array(self.full_pcd.points)[r[:self.num_points]])
        self.observed_cloud.normals = o3d.utility.Vector3dVector(np.array(self.full_pcd.normals)[r[:self.num_points]])

    def send_pcd_to_device(self):
        self.gt_gpu = torch.Tensor(np.array(self.observed_cloud.points))[None, :].to(self.device)
        self.gt_gpu_normals = torch.Tensor(np.array(self.observed_cloud.normals))[None, :].to(self.device)

    def get_single_boxes(self, path):
        self.single_boxes = json.load(open(osp.join(path, "single_bboxes.json"), "r"))["bbox"]
        self.single_boxes = [Bbox(box, idx, is_single=True, device=self.device)
                             for (idx, box) in enumerate(self.single_boxes)]
        for i, box in enumerate(self.single_boxes):
            self.single_boxes[i].idx = i

    def get_pair_boxes(self, path):
        self.pair_boxes = json.load(open(osp.join(path, "pair_bboxes.json"), "r"))["bbox"]
        self.pair_boxes = [Bbox(box, idx + len(self.single_boxes), is_single=False, device=self.device)
                           for (idx, box) in enumerate(self.pair_boxes)]
        for i, box in enumerate(self.pair_boxes):
            self.pair_boxes[i].idx = i + len(self.single_boxes)

    def merge_boxes(self):
        return self.single_boxes + self.pair_boxes

    def show_hypotheses(self, color=[0, 1, 0], thickness=0.03, priors=None):
        """
        Shows the 3D pointcloud and set of bounding boxes
        :param color: (Tuple(3)) RGB [0,1] of the bounding boxes
        :param thickness: (float) thickness of the bounding boxes edges
        """

        def p2t(prior):
            if isinstance(prior, torch.Tensor):
                prior = prior.cpu().numpy()
            return thickness * (1 - np.exp(-prior))

        if priors is not None:
            boxes_single = bboxes_to_thickboxes([box.box_points for box in self.single_boxes],
                                                thickness=[p2t(priors[box.idx]) for box in self.single_boxes], color=[1, 0, 0])
            boxes_pair = bboxes_to_thickboxes([box.box_points for box in self.pair_boxes],
                                              thickness=[p2t(priors[box.idx]) for box in self.pair_boxes], color=color)
        else:
            boxes_single = bboxes_to_thickboxes([box.box_points for box in self.single_boxes],
                                                thickness=thickness, color=[1, 0, 0])
            boxes_pair = bboxes_to_thickboxes([box.box_points for box in self.pair_boxes],
                                              thickness=thickness, color=color)
        boxes_to_draw = [boxes_single, boxes_pair]

        if self.full_pcd is not None:
            o3d.visualization.draw_geometries([self.full_pcd, *boxes_to_draw])
        else:
            o3d.visualization.draw_geometries(boxes_to_draw)

    def show_selection(self, selected_hypotheses, color=[0, 1, 0], thickness=0.01):
        """
        Show a subset of hypotheses on top of the original point cloud
        :param selected_hypotheses: List of Bbox
        :param color: (Tuple(3)) RGB [0,1] of the bounding boxes
        :param thickness: (float) thickness of the bounding boxes edges
        :return:
        """
        FULL_PCD = self.full_pcd

        def swap_geometry(vis):
            swap_geometry.CURRENT_IS_PCD = not swap_geometry.CURRENT_IS_PCD
            if swap_geometry.CURRENT_IS_PCD:
                if FULL_PCD is not None:
                    vis.remove_geometry(FULL_PCD, False)
                if single_boxes_mesh is not None:
                    vis.add_geometry(single_boxes_mesh, False)
                if pair_boxes_mesh is not None:
                    vis.add_geometry(pair_boxes_mesh, False)
            else:
                if FULL_PCD is not None:
                    vis.add_geometry(FULL_PCD, False)
                if single_boxes_mesh is not None:
                    vis.remove_geometry(single_boxes_mesh, False)
                if pair_boxes_mesh is not None:
                    vis.remove_geometry(pair_boxes_mesh, False)

            time.sleep(.25)

        single_boxes = []
        pair_boxes = []
        for proposal in selected_hypotheses:
            if proposal.is_single:
                single_boxes.append(proposal.box_points)
            else:
                pair_boxes.append(proposal.box_points)

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        single_boxes_mesh = None
        pair_boxes_mesh = None

        if len(single_boxes) != 0:
            single_boxes_linemesh = bboxes_to_thickboxes(single_boxes, thickness=thickness, color=[1, 0, 0])
            single_boxes_mesh = boxes_to_mesh_np(single_boxes).paint_uniform_color([0, 0, 1])
            single_boxes_mesh = single_boxes_mesh.compute_triangle_normals()
            single_boxes_mesh = single_boxes_mesh.compute_vertex_normals()
            vis.add_geometry(single_boxes_linemesh)

        if len(pair_boxes) != 0:
            pair_boxes_linemesh = bboxes_to_thickboxes(pair_boxes, thickness=thickness, color=color)
            pair_boxes_mesh = boxes_to_mesh_np(pair_boxes).paint_uniform_color([1, 0, 0])
            pair_boxes_mesh = pair_boxes_mesh.compute_triangle_normals()
            pair_boxes_mesh = pair_boxes_mesh.compute_vertex_normals()
            vis.add_geometry(pair_boxes_linemesh)

        if self.full_pcd is not None:
            vis.add_geometry(self.full_pcd)

        swap_geometry.CURRENT_IS_PCD = False
        vis.register_key_callback(ord("S"), swap_geometry)

        vis.run()
        vis.destroy_window()

    def load_and_show(self, path, color=[0, 1, 0], thickness=0.01):
        """
        Loads and shows a set of bounding saved in json Bbox format
        :param path: (str) Path to JSON file
        :param color: (Tuple(3)) RGB [0,1] of the bounding boxes
        :param thickness: (float) thickness of the bounding boxes edges
        :return:
        """
        with open(path, "r") as f:
            solution = json.load(f)["bbox"]
        o3d.visualization.draw_geometries([self.full_pcd,
                                           bboxes_to_thickboxes([box_pts for box_pts in solution],
                                                                thickness=thickness,
                                                                color=color)])
        return solution

    def show_observation(self):
        """
        Show the sub sampled point cloud
        :return:
        """
        o3d.visualization.draw_geometries([self.observed_cloud])

    def save_solution(self, solution, path):
        """
        Save solution in a Json file
        :param solution : (List of Bbox) the list of bounding boxes
        :param path : (str) path to Json file
        :return:
        """
        out_points = [box.box_points.tolist() for box in solution]
        with open(path, "w") as f:
            json.dump({"bbox": out_points}, f)

    #### CALLBACKS ####
    @staticmethod
    def switch_to_boxmesh(vis):
        vis.update_geometry()
