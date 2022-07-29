import torch
import numpy as np
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops import sample_points_from_meshes

from lib.box_utils import boxes_to_mesh_p3d, boxes_to_mesh_np
from lib.chamfer_distances import chamfer_distance_normals_match as chamfer_distance
from lib.visualization_utils import bboxes_to_thickboxes
import open3d as o3d

class Scoring:
    def __init__(self, cfg, observer, device=torch.device("cpu")):
        self.observer = observer
        self.cfg = cfg
        self.device = device
        self.num_points = self.cfg["num_box_points"]
        self.lambda_normals = self.cfg["lambda_normals"]
        self.lambda_single = self.cfg["lambda_single"]
        self.sampling_density = self.cfg["sampling_density"]
        self.mask_available = None


    def sample_from_proposals(self, hypotheses, device=torch.device("cuda:0")):
        box_mesh = boxes_to_mesh_p3d([box.box_points for box in hypotheses], device)
        if self.sampling_density > 0:
            faces = box_mesh.faces_packed()
            verts = box_mesh.verts_packed()
            areas, _ = mesh_face_areas_normals(verts, faces)
            num_points = int(max(1, min(self.num_points, self.sampling_density * areas.sum())))
        else:
            num_points = self.num_points
        return sample_points_from_meshes(box_mesh, num_points, return_normals=True)

    def create_synth_linemesh(self, hypotheses, color=[0, 1, 0]):
        self.box_linemesh = bboxes_to_thickboxes(hypotheses, thickness=0.05, color=color)

    def show(self, hypotheses):
        self.create_synth_linemesh(hypotheses)
        o3d.visualization.draw_geometries([self.synth_cloud, self.box_linemesh])

    def chamfer_loss(self, hypotheses=None):
        """
        Chamfer loss criterion
        :param gt_points: target 3D points
        :param gt_normals: target 3D normals
        :param hypotheses: List of Bbox
        :return: 3D chamfer loss
        """
        if len(hypotheses) > 0:
            self.synth_cloud, self.synth_normals = self.sample_from_proposals(hypotheses, device=self.device)
        else:
            return np.infty

        with torch.no_grad():
            chamfer_dist, chamfer_normals = chamfer_distance(self.observer.gt_gpu, self.synth_cloud,
                                                             x_normals=self.observer.gt_gpu_normals,
                                                             y_normals=self.synth_normals,
                                                             batch_reduction="mean",
                                                             truncation=self.cfg["chamfer_clip"],
                                                             normals_weight=self.cfg["normals_weight"],
                                                             completion_weight=self.cfg["completion_weight"])

            chamfer_final = chamfer_dist + self.lambda_normals * chamfer_normals
            return chamfer_final.mean()


    def scoring_loss(self, hypotheses=None):
        """
        :param gt_points:
        :param gt_normals:
        :param hypotheses:
        :return: scoring loss
        """
        if hypotheses is None or len(hypotheses) == 0:
            return np.inf

        loss = self.chamfer_loss(hypotheses)
        return loss