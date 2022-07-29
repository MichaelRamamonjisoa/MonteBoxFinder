import numpy as np
import open3d as o3d

import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals


def box_from_xyz_lwh_uvw(center, size, uvw):
    points = []

    points.append(center + 0.5 * np.diag([-1, -1, -1]) * size * uvw)
    points.append(center + 0.5 * np.diag([1, -1, -1]) * size * uvw)
    points.append(center + 0.5 * np.diag([1, 1, -1]) * size * uvw)
    points.append(center + 0.5 * np.diag([-1, 1, -1]) * size * uvw)
    points.append(center + 0.5 * np.diag([-1, 1, 1]) * size * uvw)
    points.append(center + 0.5 * np.diag([-1, -1, 1]) * size * uvw)
    points.append(center + 0.5 * np.diag([1, -1, 1]) * size * uvw)
    points.append(center + 0.5 * np.diag([1, 1, 1]) * size * uvw)

    return points


def box_from_uvw_proj(origin, proj, uvw):
    m = np.min(proj, axis=1)
    M = np.max(proj, axis=1)

    points = []
    points.append(uvw @ (np.array([m[0], m[1], m[2]]).T - origin) + origin)
    points.append(uvw @ (np.array([M[0], m[1], m[2]]).T - origin) + origin)
    points.append(uvw @ (np.array([M[0], M[1], m[2]]).T - origin) + origin)
    points.append(uvw @ (np.array([m[0], M[1], m[2]]).T - origin) + origin)
    points.append(uvw @ (np.array([m[0], M[1], M[2]]).T - origin) + origin)
    points.append(uvw @ (np.array([m[0], m[1], M[2]]).T - origin) + origin)
    points.append(uvw @ (np.array([M[0], m[1], M[2]]).T - origin) + origin)
    points.append(uvw @ (np.array([M[0], M[1], M[2]]).T - origin) + origin)

    return points

def bbox_mesh_from_vertices(box):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(box))
    np_triangles = np.array([[3, 1, 0], [3, 2, 1],
                             [1, 2, 6], [2, 7, 6],
                             [4, 5, 6], [6, 7, 4],
                             [0, 1, 6], [0, 6, 5],
                             [0, 4, 3], [0, 5, 4],
                             [2, 3, 7], [3, 4, 7], ]).astype(np.int32)
    mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
    mesh.compute_vertex_normals()
    return mesh


def boxes_to_mesh(boxes):
    mesh = o3d.geometry.TriangleMesh()
    vertices = []
    triangles = []
    np_triangles = np.array([[3, 1, 0], [3, 2, 1],
                             [1, 2, 6], [2, 7, 6],
                             [4, 5, 6], [6, 7, 4],
                             [0, 1, 6], [0, 6, 5],
                             [0, 4, 3], [0, 5, 4],
                             [2, 3, 7], [3, 4, 7], ]).astype(np.int32)
    for box in boxes:
        vertices += box
        triangles += np_triangles.tolist()
        np_triangles += 8

    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    # mesh.compute_vertex_normals()
    return mesh


def boxes_to_mesh_np(boxes):
    # much faster than the non numpy optimized version
    mesh = o3d.geometry.TriangleMesh()
    triangles_per_box = 12
    vertices_per_box = 8
    num_boxes = len(boxes)
    offsets = np.arange(num_boxes) * vertices_per_box

    np_triangles = np.array([[3, 1, 0], [3, 2, 1],
                             [1, 2, 6], [2, 7, 6],
                             [4, 5, 6], [6, 7, 4],
                             [0, 1, 6], [0, 6, 5],
                             [0, 4, 3], [0, 5, 4],
                             [2, 3, 7], [3, 4, 7], ]).astype(np.int32)

    mesh.triangles = o3d.utility.Vector3iVector(
        np.tile(np_triangles, (num_boxes, 1)) + np.tile(np.repeat(offsets, triangles_per_box), (3, 1)).T)
    mesh.vertices = o3d.utility.Vector3dVector(np.concatenate(boxes))
    mesh.compute_vertex_normals()
    return mesh


def boxes_to_mesh_p3d(boxes, device=torch.device("cpu")):
    # much faster than the non numpy optimized version
    triangles_per_box = 12
    vertices_per_box = 8
    num_boxes = len(boxes)
    offsets = torch.arange(num_boxes, device=device) * vertices_per_box

    np_triangles = torch.tensor([[3, 1, 0], [3, 2, 1],
                                 [1, 2, 6], [2, 7, 6],
                                 [4, 5, 6], [6, 7, 4],
                                 [0, 1, 6], [0, 6, 5],
                                 [0, 4, 3], [0, 5, 4],
                                 [2, 3, 7], [3, 4, 7]], device=device)
    triangles = np_triangles.repeat(num_boxes, 1) + torch.repeat_interleave(offsets,
                                                                            repeats=triangles_per_box).repeat(3, 1).T
    vertices = torch.from_numpy(np.asarray(boxes)).to(device).float()
    vertices = torch.reshape(vertices, (-1, 3))
    return Meshes(verts=[vertices], faces=[triangles])


def sample_points_from_boxes(boxes, num_points, return_normals=False, device=torch.device("cpu")):
    mesh = boxes_to_mesh_p3d(boxes, device)
    return sample_points_from_meshes(mesh, num_points, return_normals)


def sample_points_from_boxes_with_density(boxes, density, max_points, return_normals=False, device=torch.device("cpu")):
    mesh = boxes_to_mesh_p3d(boxes, device)
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    areas, _ = mesh_face_areas_normals(verts, faces)
    num_points_from_density = density * areas.sum()
    return sample_points_from_meshes(mesh, min(max_points, num_points_from_density), return_normals)


def test_one_vs_N_boxes_2(query_box, boxes, tolerance=-1):
    xyz_A = np.array(query_box)
    delta_A = (xyz_A[2:5] - xyz_A[1:4])
    xyz_B = np.array(boxes)
    delta_B = (xyz_B[:, 2:5] - xyz_B[:, 1:4])
    LWH_A = np.linalg.norm(delta_A, axis=1, keepdims=True)
    LWH_B = np.linalg.norm(delta_B, axis=2, keepdims=True)

    normals_A = delta_A / LWH_A
    normals_B = delta_B / LWH_B

    LWH_B = np.reshape(LWH_B, (-1, 3))
    normals_B = np.reshape(normals_B, (-1, 3))
    xyz_B = np.reshape(xyz_B, (-1, 3))

    # concat A to B seperately
    AcatB = np.concatenate([normals_A, -normals_A, normals_B, -normals_B])

    dot_product = np.concatenate([normals_A, -normals_A, normals_B, -normals_B]) @ (np.concatenate([xyz_A, xyz_B], 0).T)
    dot_product = np.reshape(dot_product, (-1,))

    #     N x (12 x 3) * (3 x (8 + 8)) x N

    # min projection of Bs on A
    projAonA = np.reshape(dot_product[:6, :8])
    max_projA = np.max(dot_product[:, :8], 1)
    min_projA = np.min(dot_product[:, :8], 1)

    dot_product = dot_product[:, 8:]
    max_projB = np.max(dot_product[:, 8:], 1)
    min_projB = np.min(dot_product[:, 8:], 1)

    if (np.any(np.logical_or(max_projA < min_projB, min_projA > max_projB))):
        return False
    elif tolerance < 0:
        return True

    # there is some overlap, if it is not significant we consider intersection to be NULL
    overlap = np.abs(np.max([min_projA, min_projB], axis=0) - np.min([max_projA, max_projB], axis=0))
    # length_A = max_projA - min_projA
    # length_B = max_projB - min_projB
    length_A = np.sum(LWH_A ** 2)
    length_B = np.sum(LWH_B ** 2)

    return np.any(overlap > tolerance * (length_A + length_B) / 2)


def test_one_vs_N_boxes_1(query_box, boxes, tolerance=-1):
    for box in boxes:
        if test_intersection(np.array(query_box), np.array(box), tolerance):
            return True


if __name__ == "__main__":
    box_A = np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 1.0, 1.0],
                      [0.0, 0.0, 1.0],
                      [1.0, 0.0, 1.0],
                      [1.0, 1.0, 1.0]])
    box_B = box_A + 0.98
    assert test_intersection(box_A, box_B)
    assert (not test_intersection(box_A, box_B, 0.021))
