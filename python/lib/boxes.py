import numpy as np
import torch
from lib.iou_box3d import box3d_overlap_sampling_batched as box3d_overlap
from lib.iou_box3d import box_volume
from ordered_set import OrderedSet
import copy

class Bbox:
    def __init__(self, box_points, idx, is_single=False, device=torch.device("cpu")):
        self.box_points = np.array(box_points)
        xyz = (self.box_points[2:5] - self.box_points[1:4])
        LWH = np.linalg.norm(xyz, axis=1)
        self.volume = np.abs(np.prod(LWH))
        self.intersect_list = set()
        self.idx = idx
        self.device = device
        self.incompatible_boxes = OrderedSet()
        self.compatible_boxes = OrderedSet()
        self.is_single = is_single

    @staticmethod
    def cgal_to_p3d_corners(box_points_array):
        order = [0, 1, 2, 3, 5, 6, 7, 4]
        return np.array(copy.copy(box_points_array)[order])

    def update_intersections(self, boxes, tolerance=-1):
        if len(boxes)==0:
            return True
        boxes_set = torch.stack([torch.Tensor(self.cgal_to_p3d_corners(box.box_points)) for box in boxes])
        box_query = torch.Tensor(self.cgal_to_p3d_corners(self.box_points)).unsqueeze(0)
        _, IoUs = box3d_overlap(boxes_set, box_query)
        for i in range(len(boxes)):
            if boxes[i].idx == self.idx or boxes[i].idx in self.intersect_list:
                continue
            if IoUs[i] > tolerance:
                self.intersect_list.add(boxes[i].idx)

    def compatible(self, boxes, tolerance=0):
        if len(boxes)==0:
            return True, 0
        box_query = torch.Tensor(self.cgal_to_p3d_corners(self.box_points)).to(self.device)
        vol_query = box_volume(box_query[None])
        boxes_set = torch.stack([torch.Tensor(self.cgal_to_p3d_corners(box.box_points)) for box in boxes]).to(self.device)
        vols_set = box_volume(boxes_set)

        inters, _ = box3d_overlap(box_query[None], boxes_set)
        iou = inters / torch.where(vol_query > vols_set, vols_set, vol_query)
        true_iou = inters / (vols_set + vol_query - inters)
        return iou < tolerance, true_iou

    def measure_iou(self, boxes):
        if len(boxes)==0:
            return torch.zeros((1,))
        box_query = torch.Tensor(self.cgal_to_p3d_corners(self.box_points)).to(self.device)
        vol_query = box_volume(box_query[None])  # (1,)
        boxes_set = torch.stack([torch.Tensor(self.cgal_to_p3d_corners(box.box_points)) for box in boxes]).to(
            self.device)  # (N, 1)
        vols_set = box_volume(boxes_set)  # (N, 1)
        inters, _ = box3d_overlap(box_query[None], boxes_set)  # ((N, 1), (N, 1))
        return inters / (vols_set + vol_query - inters)  # (N, 1)
