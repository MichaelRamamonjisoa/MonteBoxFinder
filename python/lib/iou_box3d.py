# Large portions of this code is taken from Meta Platform's code on Pytorch3d
# https://github.com/facebookresearch/pytorch3d/blob/main/tests/test_iou_box3d.py
# and
# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/iou_box3d.py


from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from pytorch3d import _C
from torch.autograd import Function


# -------------------------------------------------- #
#                  CONSTANTS                         #
# -------------------------------------------------- #
"""
_box_planes and _box_triangles define the 4- and 3-connectivity
of the 8 box corners.
_box_planes gives the quad faces of the 3D box
_box_triangles gives the triangle faces of the 3D box
"""
_box_planes = [
    [0, 1, 2, 3],
    [3, 2, 6, 7],
    [0, 1, 5, 4],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [4, 5, 6, 7],
]
_box_triangles = [
    [0, 1, 2],
    [0, 2, 3],
    [4, 5, 6],
    [4, 6, 7],
    [1, 5, 6],
    [1, 6, 2],
    [0, 4, 7],
    [0, 7, 3],
    [3, 2, 6],
    [3, 6, 7],
    [0, 1, 5],
    [0, 5, 4],
]


def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-6) -> None:
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    # pyre-fixme[16]: `boxes` has no attribute `index_select`.
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)
    if not (mat1.bmm(mat2).abs() < eps).all().item():
        msg = "Plane vertices are not coplanar"
        raise ValueError(msg)

    return


def _check_nonzero(boxes: torch.Tensor, eps: float = 1e-6) -> None:
    """
    Checks that the sides of the box have a non zero area
    """
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    # pyre-fixme[16]: `boxes` has no attribute `index_select`.
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    if (face_areas < eps).any().item():
        msg = "Planes have zero areas"
        raise ValueError(msg)

    return


def box3d_overlap(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the intersection of 3D boxes1 and boxes2.

    Inputs boxes1, boxes2 are tensors of shape (B, 8, 3)
    (where B doesn't have to be the same for boxes1 and boxes1),
    containing the 8 corners of the boxes, as follows:

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)


    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:

    box_corner_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    Args:
        boxes1: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
        boxes2: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
    Returns:
        vol: (N, M) tensor of the volume of the intersecting convex shapes
        iou: (N, M) tensor of the intersection over union which is
            defined as: `iou = vol / (vol1 + vol2 - vol)`
    """
    if not all((8, 3) == box.shape[1:] for box in [boxes1, boxes2]):
        raise ValueError("Each box in the batch must be of shape (8, 3)")

    _check_coplanar(boxes1, eps)
    _check_coplanar(boxes2, eps)
    _check_nonzero(boxes1, eps)
    _check_nonzero(boxes2, eps)

    # TODO
    iou = 0
    vol = 0



    return vol, iou

# -------------------------------------------------- #
#               MAIN: BOX3D_OVERLAP                  #
# -------------------------------------------------- #

# -------------------------------------------------- #
#       HELPER FUNCTIONS FOR EXACT SOLUTION          #
# -------------------------------------------------- #


def get_tri_verts(boxes: torch.Tensor) -> torch.Tensor:
    """
    Return the vertex coordinates forming the triangles of the box.
    The computation here resembles the Meshes data structure.
    But since we only want this tiny functionality, we abstract it out.
    Args:
        boxes: tensor of shape (N, 8, 3)
    Returns:
        tri_verts: tensor of shape (N, 12, 3, 3)
    """
    device = boxes.device
    faces = torch.tensor(_box_triangles, device=device, dtype=torch.int64)  # (12, 3)
    tri_verts = boxes[:, faces]  # (N, 12, 3, 3)
    return tri_verts


def get_plane_verts(boxes: torch.Tensor) -> torch.Tensor:
    """
    Return the vertex coordinates forming the planes of the box.
    The computation here resembles the Meshes data structure.
    But since we only want this tiny functionality, we abstract it out.
    Args:
        boxes: tensor of shape (N, 8, 3)
    Returns:
        plane_verts: tensor of shape (N, 6, 4, 3)
    """
    device = boxes.device
    faces = torch.tensor(_box_planes, device=device, dtype=torch.int64)  # (6, 4)
    plane_verts = boxes[:, faces]  # (N, 6, 4, 3)
    return plane_verts


def box_planar_dir(boxes: torch.Tensor, eps=1e-5) -> torch.Tensor:
    """
    Finds the unit vector n which is perpendicular to each plane in the box
    and points towards the inside of the box.
    The planes are defined by `_box_planes`.
    Since the shape is convex, we define the interior to be the direction
    pointing to the center of the shape.
    Args:
       boxes: tensor of shape (N, 8, 3) of the vertices of the 3D box
    Returns:
       n: tensor of shape (N, 6, 3) of the unit vectors orthogonal to the faces pointing
          towards the interior of the shape
    """
    assert boxes.shape[1] == 8 and boxes.shape[2] == 3
    N = boxes.shape[0]

    # center point of each box
    ctr = boxes.mean(1).view(N, 1, 3)

    verts = get_plane_verts(boxes)  # (N, 6, 4, 3)

    v0, v1, v2, v3 = verts.unbind(2)  # each vertex of each plane (of each box) (N, 6, 3)

    # We project the ctr on the plane defined by (v0, v1, v2, v3)
    # We define e0 to be the edge connecting (v1, v0)
    # We define e1 to be the edge connecting (v2, v0)
    # And n is the cross product of e0, e1, either pointing "inside" or not.
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    n = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check all verts are coplanar for each box
    # if not ((v3 - v0).unsqueeze(1).bmm(n.unsqueeze(2)).abs() < eps).all().item():
    if not (torch.einsum("bpd,bpd->bp", v3-v0, n).abs() < eps).all().item():
        msg = "Plane vertices are not coplanar"
        raise ValueError(msg)

    # Check all faces have non zero area
    area1 = torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1) / 2
    area2 = torch.cross(v3 - v0, v2 - v0, dim=-1).norm(dim=-1) / 2
    if (area1 < eps).any().item() or (area2 < eps).any().item():
        msg = "Planes have zero areas"
        raise ValueError(msg)

    # We can write:  `ctr = v0 + a * e0 + b * e1 + c * n`, (1).
    # With <e0, n> = 0 and <e1, n> = 0, where <.,.> refers to the dot product,
    # since that e0 is orthogonal to n. Same for e1.
    """
    # Below is how one would solve for (a, b, c)
    # Solving for (a, b)
    numF = verts.shape[0]
    A = torch.ones((numF, 2, 2), dtype=torch.float32, device=device)
    B = torch.ones((numF, 2), dtype=torch.float32, device=device)
    A[:, 0, 1] = (e0 * e1).sum(-1)
    A[:, 1, 0] = (e0 * e1).sum(-1)
    B[:, 0] = ((ctr - v0) * e0).sum(-1)
    B[:, 1] = ((ctr - v1) * e1).sum(-1)
    ab = torch.linalg.solve(A, B)  # (numF, 2)
    a, b = ab.unbind(1)
    # solving for c
    c = ((ctr - v0 - a.view(numF, 1) * e0 - b.view(numF, 1) * e1) * n).sum(-1)
    """
    # Since we know that <e0, n> = 0 and <e1, n> = 0 (e0 and e1 are orthogonal to n),
    # the above solution is equivalent to
    c = ((ctr - v0) * n).sum(-1)
    # If c is negative, then we revert the direction of n such that n points "inside"
    negc = c < 0.0
    n[negc] *= -1.0
    # c[negc] *= -1.0
    # Now (a, b, c) is the solution to (1)

    return n


def tri_verts_area(tri_verts: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of the triangle faces in tri_verts
    Args:
        tri_verts: tensor of shape (T, 3, 3)
    Returns:
        areas: the area of the triangles (T, 1)
    """
    add_dim = False
    if tri_verts.ndim == 2:
        tri_verts = tri_verts.unsqueeze(0)
        add_dim = True

    v0, v1, v2 = tri_verts.unbind(1)
    areas = torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1) / 2.0

    if add_dim:
        areas = areas[0]
    return areas


def box_volume(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the volume of each box in boxes.
    The volume of each box is the sum of all the tetrahedrons
    formed by the faces of the box. The face of the box is the base of
    that tetrahedron and the center point of the box is the apex.
    In other words, vol(box) = sum_i A_i * d_i / 3,
    where A_i is the area of the i-th face and d_i is the
    distance of the apex from the face.
    We use the equivalent dot/cross product formulation.
    Read https://en.wikipedia.org/wiki/Tetrahedron#Volume
    Args:
        box: tensor of shape (N, 8, 3) containing the vertices
            of the 3D box
    Returns:
        vols: the volume of the box
    """
    assert boxes.shape[1] == 8 and boxes.shape[2] == 3

    N = boxes.shape[0]
    # Compute the center point of each box
    ctr = boxes.mean(1).view(N, 1, 1, 3)

    # Extract the coordinates of the faces for each box
    tri_verts = get_tri_verts(boxes)
    # Set the origin of the coordinate system to coincide
    # with the apex of the tetrahedron to simplify the volume calculation
    # See https://en.wikipedia.org/wiki/Tetrahedron#Volume
    tri_verts = tri_verts - ctr

    # Compute the volume of each box using the dot/cross product formula
    vols = torch.sum(
        tri_verts[:, :, 0] * torch.cross(tri_verts[:, :, 1], tri_verts[:, :, 2], dim=-1),
        dim=-1,
    )
    vols = (vols.abs() / 6.0).sum(-1)

    return vols


def coplanar_tri_faces(tri1: torch.Tensor, tri2: torch.Tensor, eps: float = 1e-6):
    """
    Determines whether two triangle faces in 3D are coplanar
    Args:
        tri1: tensor of shape (3, 3) of the vertices of the 1st triangle
        tri2: tensor of shape (3, 3) of the vertices of the 2nd triangle
    Returns:
        is_coplanar: bool
    """
    v0, v1, v2 = tri1.unbind(0)
    e0 = F.normalize(v1 - v0, dim=0)
    e1 = F.normalize(v2 - v0, dim=0)
    e2 = F.normalize(torch.cross(e0, e1), dim=0)

    coplanar2 = torch.zeros((3,), dtype=torch.bool, device=tri1.device)
    for i in range(3):
        if (tri2[i] - v0).dot(e2).abs() < eps:
            coplanar2[i] = 1
    coplanar2 = coplanar2.all()
    return coplanar2


def is_inside(
    plane: torch.Tensor,
    n: torch.Tensor,
    points: torch.Tensor,
    return_proj: bool = True,
    eps: float = 1e-6,
):
    """
    Computes whether point is "inside" the plane.
    The definition of "inside" means that the point
    has a positive component in the direction of the plane normal defined by n.
    For example,
                  plane
                    |
                    |         . (A)
                    |--> n
                    |
         .(B)       |
    Point (A) is "inside" the plane, while point (B) is "outside" the plane.
    Args:
      plane: tensor of shape (4,3) of vertices of a box plane
      n: tensor of shape (3,) of the unit "inside" direction on the plane
      points: tensor of shape (P, 3) of coordinates of a point
      return_proj: bool whether to return the projected point on the plane
    Returns:
      is_inside: bool of shape (P) of whether point is inside
      p_proj: tensor of shape (P, 3) of the projected point on plane
    """
    #TODO batchify this!!
    device = plane.device
    v0, v1, v2, v3 = plane
    e0 = F.normalize(v1 - v0, dim=1)
    e1 = F.normalize(v2 - v0, dim=1)
    if not torch.allclose(e0.dot(n), torch.zeros((1,), device=device), atol=eps):
        raise ValueError("Input n is not perpendicular to the plane")
    if not torch.allclose(e1.dot(n), torch.zeros((1,), device=device), atol=eps):
        raise ValueError("Input n is not perpendicular to the plane")

    add_dim = False
    if points.ndim == 1:
        points = points.unsqueeze(0)
        add_dim = True

    assert points.shape[1] == 3
    # Every point p can be written as p = v0 + a e0 + b e1 + c n

    # If return_proj is True, we need to solve for (a, b)
    p_proj = None
    if return_proj:
        # solving for (a, b)
        A = torch.tensor(
            [[1.0, e0.dot(e1)], [e0.dot(e1), 1.0]], dtype=torch.float32, device=device
        )
        B = torch.zeros((2, points.shape[0]), dtype=torch.float32, device=device)
        B[0, :] = torch.sum((points - v0.view(1, 3)) * e0.view(1, 3), dim=-1)
        B[1, :] = torch.sum((points - v0.view(1, 3)) * e1.view(1, 3), dim=-1)

        ab = A.inverse() @ B  # (2, P)
        p_proj = v0.view(1, 3) + ab.transpose(0, 1) @ torch.stack((e0, e1), dim=0)

    # solving for c
    # c = (point - v0 - a * e0 - b * e1).dot(n)
    c = torch.sum((points - v0.view(1, 3)) * n.view(1, 3), dim=-1)
    ins = c > -eps

    if add_dim:
        assert p_proj.shape[0] == 1
        p_proj = p_proj[0]

    return ins, p_proj


def are_inside(
    planes: torch.Tensor,
    ns: torch.Tensor,
    points: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Computes whether point is "inside" the plane.
    The definition of "inside" means that the point
    has a positive component in the direction of the plane normal defined by n.
    For example,
                  plane
                    |
                    |         . (A)
                    |--> n
                    |
         .(B)       |
    Point (A) is "inside" the plane, while point (B) is "outside" the plane.
    Args:
      planes: tensor of shape (N,6,4,3) of the 4 vertices of the 6 planes of each box
      ns: tensor of shape (N,6,3) of the unit "inside" direction of each of the 6 planes
      points: tensor of shape (P, 3) of coordinates of a point
    Returns:
      is_inside: bool of shape (N,P,6) of whether point is inside
    """

    device = planes.device
    N = planes.shape[0]
    assert ns.shape[0] == N
    assert ns.shape[1] == 6

    v0, v1, v2, v3 = planes.unbind(2)  # each vertex of the 6 planes of each box (N, 6, 3)

    e0 = F.normalize(v1 - v0, dim=-1) # (N, 6, 3)
    e1 = F.normalize(v2 - v0, dim=-1) # (N, 6, 3)

    if not torch.allclose((e0 * ns).sum(-1), torch.zeros((N, 6), device=device), atol=eps):
        raise ValueError("Input n is not perpendicular to the plane")
    if not torch.allclose((e1 * ns).sum(-1), torch.zeros((N, 6), device=device), atol=eps):
        raise ValueError("Input n is not perpendicular to the plane")

    assert points.shape[-1] == 3

    # c = torch.sum((points[None] - v0[:, None]) * ns[:, None], dim=-1)
    c = torch.sum((points[None, :, None] - v0[:, None]) * ns[:, None], dim=-1)
    ins = c > -eps

    return ins



def plane_edge_point_of_intersection(plane, n, p0, p1):
    """
    Finds the point of intersection between a box plane and
    a line segment connecting (p0, p1).
    The plane is assumed to be infinite long.
    Args:
      plane: tensor of shape (4, 3) of the coordinates of the vertices defining the plane
      n: tensor of shape (3,) of the unit direction perpendicular on the plane
          (Note that we could compute n but since it's computed in the main
          body of the function, we save time by feeding it in. For the purpose
          of this function, it's not important that n points "inside" the shape.)
      p0, p1: tensors of shape (3,), (3,)
    Returns:
      p: tensor of shape (3,) of the coordinates of the point of intersection
      a: scalar such that p = p0 + a*(p1-p0)
    """
    # The point of intersection can be parametrized
    # p = p0 + a (p1 - p0) where a in [0, 1]
    # We want to find a such that p is on plane
    # <p - v0, n> = 0
    v0, v1, v2, v3 = plane
    a = -(p0 - v0).dot(n) / (p1 - p0).dot(n)
    p = p0 + a * (p1 - p0)
    return p, a


"""
The three following functions support clipping a triangle face by a plane.
They contain the following cases: (a) the triangle has one point "outside" the plane and
(b) the triangle has two points "outside" the plane.
This logic follows the logic of clipping triangles when they intersect the image plane while
rendering.
"""


def clip_tri_by_plane_oneout(
    plane: torch.Tensor,
    n: torch.Tensor,
    vout: torch.Tensor,
    vin1: torch.Tensor,
    vin2: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Case (a).
    Clips triangle by plane when vout is outside plane, and vin1, vin2, is inside
    In this case, only one vertex of the triangle is outside the plane.
    Clip the triangle into a quadrilateral, and then split into two triangles
    Args:
        plane: tensor of shape (4, 3) of the coordinates of the vertices forming the plane
        n: tensor of shape (3,) of the unit "inside" direction of the plane
        vout, vin1, vin2: tensors of shape (3,) of the points forming the triangle, where
            vout is "outside" the plane and vin1, vin2 are "inside"
    Returns:
        verts: tensor of shape (4, 3) containing the new vertices formed after clipping the
            original intersecting triangle (vout, vin1, vin2)
        faces: tensor of shape (2, 3) defining the vertex indices forming the two new triangles
            which are "inside" the plane formed after clipping
    """
    device = plane.device
    # point of intersection between plane and (vin1, vout)
    pint1, a1 = plane_edge_point_of_intersection(plane, n, vin1, vout)
    assert a1 >= -eps and a1 <= 1.0 + eps, a1
    # point of intersection between plane and (vin2, vout)
    pint2, a2 = plane_edge_point_of_intersection(plane, n, vin2, vout)
    assert a2 >= -eps and a2 <= 1.0 + eps, a2

    verts = torch.stack((vin1, pint1, pint2, vin2), dim=0)  # 4x3
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3]], dtype=torch.int64, device=device
    )  # 2x3
    return verts, faces


def clip_tri_by_plane_twoout(
    plane: torch.Tensor,
    n: torch.Tensor,
    vout1: torch.Tensor,
    vout2: torch.Tensor,
    vin: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Case (b).
    Clips face by plane when vout1, vout2 are outside plane, and vin1 is inside
    In this case, only one vertex of the triangle is inside the plane.
    Args:
        plane: tensor of shape (4, 3) of the coordinates of the vertices forming the plane
        n: tensor of shape (3,) of the unit "inside" direction of the plane
        vout1, vout2, vin: tensors of shape (3,) of the points forming the triangle, where
            vin is "inside" the plane and vout1, vout2 are "outside"
    Returns:
        verts: tensor of shape (3, 3) containing the new vertices formed after clipping the
            original intersectiong triangle (vout, vin1, vin2)
        faces: tensor of shape (1, 3) defining the vertex indices forming
            the single new triangle which is "inside" the plane formed after clipping
    """
    device = plane.device
    # point of intersection between plane and (vin, vout1)
    pint1, a1 = plane_edge_point_of_intersection(plane, n, vin, vout1)
    assert a1 >= -eps and a1 <= 1.0 + eps, a1
    # point of intersection between plane and (vin, vout2)
    pint2, a2 = plane_edge_point_of_intersection(plane, n, vin, vout2)
    assert a2 >= -eps and a2 <= 1.0 + eps, a2

    verts = torch.stack((vin, pint1, pint2), dim=0)  # 3x3
    faces = torch.tensor(
        [
            [0, 1, 2],
        ],
        dtype=torch.int64,
        device=device,
    )  # 1x3
    return verts, faces


def clip_tri_by_plane(plane, n, tri_verts) -> Union[List, torch.Tensor]:
    """
    Clip a trianglular face defined by tri_verts with a plane of inside "direction" n.
    This function computes whether the triangle has one or two
    or none points "outside" the plane.
    Args:
       plane: tensor of shape (4, 3) of the vertex coordinates of the plane
       n: tensor of shape (3,) of the unit "inside" direction of the plane
       tri_verts: tensor of shape (3, 3) of the vertex coordiantes of the the triangle faces
    Returns:
        tri_verts: tensor of shape (K, 3, 3) of the vertex coordinates of the triangles formed
            after clipping. All K triangles are now "inside" the plane.
    """
    v0, v1, v2 = tri_verts.unbind(0)
    isin0, _ = is_inside(plane, n, v0)
    isin1, _ = is_inside(plane, n, v1)
    isin2, _ = is_inside(plane, n, v2)

    if isin0 and isin1 and isin2:
        # all in, no clipping, keep the old triangle face
        return tri_verts.view(1, 3, 3)
    elif (not isin0) and (not isin1) and (not isin2):
        # all out, delete triangle
        return []
    else:
        if isin0:
            if isin1:  # (isin0, isin1, not isin2)
                verts, faces = clip_tri_by_plane_oneout(plane, n, v2, v0, v1)
                return verts[faces]
            elif isin2:  # (isin0, not isin1, isin2)
                verts, faces = clip_tri_by_plane_oneout(plane, n, v1, v0, v2)
                return verts[faces]
            else:  # (isin0, not isin1, not isin2)
                verts, faces = clip_tri_by_plane_twoout(plane, n, v1, v2, v0)
                return verts[faces]
        else:
            if isin1 and isin2:  # (not isin0, isin1, isin2)
                verts, faces = clip_tri_by_plane_oneout(plane, n, v0, v1, v2)
                return verts[faces]
            elif isin1:  # (not isin0, isin1, not isin2)
                verts, faces = clip_tri_by_plane_twoout(plane, n, v0, v2, v1)
                return verts[faces]
            elif isin2:  # (not isin0, not isin1, isin2)
                verts, faces = clip_tri_by_plane_twoout(plane, n, v0, v1, v2)
                return verts[faces]

    # Should not be reached
    return []


def box3d_overlap_naive(box1: torch.Tensor, box2: torch.Tensor):
    """
    Computes the intersection of 3D boxes1 and boxes2.
    Inputs boxes1, boxes2 are tensors of shape (8, 3) containing
    the 8 corners of the boxes, as follows
        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)
    Args:
        box1: tensor of shape (8, 3) of the coordinates of the 1st box
        box2: tensor of shape (8, 3) of the coordinates of the 2nd box
    Returns:
        vol: the volume of the intersecting convex shape
        iou: the intersection over union which is simply
            `iou = vol / (vol1 + vol2 - vol)`
    """
    device = box1.device

    if box1.shape[0] == 1:
        box1 = box1[0]
    if box2.shape[0] == 1:
        box2 = box2[0]
    # For boxes1 we compute the unit directions n1 corresponding to quad_faces
    n1 = box_planar_dir(box1)  # (6, 3)
    # For boxes2 we compute the unit directions n2 corresponding to quad_faces
    n2 = box_planar_dir(box2)

    # We define triangle faces
    vol1 = box_volume(box1)
    vol2 = box_volume(box2)

    tri_verts1 = get_tri_verts(box1)  # (12, 3, 3)
    plane_verts1 = get_plane_verts(box1)  # (6, 4, 3)
    tri_verts2 = get_tri_verts(box2)  # (12, 3, 3)
    plane_verts2 = get_plane_verts(box2)  # (6, 4, 3)

    num_planes = plane_verts1.shape[0]  # (=6) based on our definition of planes

    # Every triangle in box1 will be compared to each plane in box2.
    # If the triangle is fully outside or fully inside, then it will remain as is
    # If the triangle intersects with the (infinite) plane, it will be broken into
    # subtriangles such that each subtriangle is either fully inside or outside the plane.

    # Tris in Box1 -> Planes in Box2
    for pidx in range(num_planes):
        plane = plane_verts2[pidx]
        nplane = n2[pidx]
        tri_verts_updated = torch.zeros((0, 3, 3), dtype=torch.float32, device=device)
        for i in range(tri_verts1.shape[0]):
            tri = clip_tri_by_plane(plane, nplane, tri_verts1[i])
            if len(tri) > 0:
                tri_verts_updated = torch.cat((tri_verts_updated, tri), dim=0)
        tri_verts1 = tri_verts_updated

    # Tris in Box2 -> Planes in Box1
    for pidx in range(num_planes):
        plane = plane_verts1[pidx]
        nplane = n1[pidx]
        tri_verts_updated = torch.zeros((0, 3, 3), dtype=torch.float32, device=device)
        for i in range(tri_verts2.shape[0]):
            tri = clip_tri_by_plane(plane, nplane, tri_verts2[i])
            if len(tri) > 0:
                tri_verts_updated = torch.cat((tri_verts_updated, tri), dim=0)
        tri_verts2 = tri_verts_updated

    # remove triangles that are coplanar from the intersection as
    # otherwise they would be doublecounting towards the volume
    # this happens only if the original 3D boxes have common planes
    # Since the resulting shape is convex and specifically composed of planar segments,
    # each planar segment can belong either on box1 or box2 but not both.
    # Without loss of generality, we assign shared planar segments to box1
    keep2 = torch.ones((tri_verts2.shape[0],), device=device, dtype=torch.bool)
    for i1 in range(tri_verts1.shape[0]):
        for i2 in range(tri_verts2.shape[0]):
            if (
                coplanar_tri_faces(tri_verts1[i1], tri_verts2[i2])
                and tri_verts_area(tri_verts1[i1]) > 1e-6
            ):
                keep2[i2] = 0
    keep2 = keep2.nonzero()[:, 0]
    tri_verts2 = tri_verts2[keep2]

    # intersecting shape
    num_faces = tri_verts1.shape[0] + tri_verts2.shape[0]
    num_verts = num_faces * 3  # V=F*3
    overlap_faces = torch.arange(num_verts).view(num_faces, 3)  # Fx3
    overlap_tri_verts = torch.cat((tri_verts1, tri_verts2), dim=0)  # Fx3x3
    overlap_verts = overlap_tri_verts.view(num_verts, 3)  # Vx3

    # the volume of the convex hull defined by (overlap_verts, overlap_faces)
    # can be defined as the sum of all the tetrahedrons formed where for each tetrahedron
    # the base is the triangle and the apex is the center point of the convex hull
    # See the math here: https://en.wikipedia.org/wiki/Tetrahedron#Volume

    # we compute the center by computing the center point of each face
    # and then averaging the face centers
    ctr = overlap_tri_verts.mean(1).mean(0)
    tetras = overlap_tri_verts - ctr.view(1, 1, 3)
    vol = torch.sum(
        tetras[:, 0] * torch.cross(tetras[:, 1], tetras[:, 2], dim=-1), dim=-1
    )
    vol = (vol.abs() / 6.0).sum()

    iou = vol / (vol1 + vol2 - vol)
    return vol, iou


# -------------------------------------------------- #
#       HELPER FUNCTIONS FOR SAMPLING SOLUTION       #
# -------------------------------------------------- #


def is_point_inside_box(box: torch.Tensor, points: torch.Tensor, eps=1e-6):
    """
    Determines whether points are inside the boxes
    Args:
        box: tensor of shape (8, 3) of the corners of the boxes
        points: tensor of shape (P, 3) of the points
    Returns:
        inside: bool tensor of shape (P,)
    """
    device = box.device
    assert points.shape[0] > 0
    P = points.shape[0]

    n = box_planar_dir(box, eps)  # (6, 3)
    box_planes = get_plane_verts(box)  # (6, 4)
    num_planes = box_planes.shape[0]  # = 6

    # a point p is inside the box if it "inside" all planes of the box
    # so we run the checks
    ins = torch.zeros((P, num_planes), device=device, dtype=torch.bool)
    for i in range(num_planes):
        is_in, _ = is_inside(box_planes[i], n[i], points, return_proj=False, eps=eps)
        ins[:, i] = is_in
    ins = ins.all(dim=1)
    return ins


def are_points_inside_boxes(boxes: torch.Tensor, points: torch.Tensor, eps=1e-6):
    """
    Determines whether points are inside the boxes
    Args:
        box: tensor of shape (N, 8, 3) of the corners of the boxes
        points: tensor of shape (P, 3) of the points
    Returns:
        inside: bool tensor of shape (N, P)
    """
    device = boxes.device
    if points.ndim == 3:
        points = points.view(-1, 3)
    # P = points.shape[1]
    assert points.shape[0] > 0
    N = boxes.shape[0]

    try:
        n = box_planar_dir(boxes, eps)  # (N, 6, 3)
    except ValueError:
        n = None
    boxes_planes = get_plane_verts(boxes)  # (N, 6, 4)
    num_planes = boxes_planes.shape[1]  # = 6

    # a point p is inside the box if it is "inside" all planes of the box
    # so we run the checks
    # ins = torch.zeros((N, P, num_planes), device=device, dtype=torch.bool)

    # checks if points are "inside" the planes that constitute the box
    if n is not None:
        ins = are_inside(boxes_planes, n, points, eps) # (N, P, num_planes)
        ins = ins.all(dim=2)
    else:
        ins = torch.zeros((1, points.shape[0]), dtype=torch.bool, device=device)
    return ins


def sample_points_within_boxes(boxes: torch.Tensor, num_samples: int = 10):
    """
    Sample points within a box defined by its 8 coordinates
    Args:
        box: tensor of shape (N, 8, 3) of the box coordinates
        num_samples: int defining the number of samples per bounding box
    Returns:
        points: (N, num_samples, 3) of points inside the box
    """
    assert boxes.shape[1] == 8 and boxes.shape[2] == 3
    N = boxes.shape[0]

    U = boxes[:, 1:2] - boxes[:, 0:1] # (N, 1, 3)
    V = boxes[:, 2:3] - boxes[:, 1:2] # (N, 1, 3)
    W = boxes[:, 4:5] - boxes[:, 0:1] # (N, 1, 3)

    UVW = torch.cat([U, V, W], dim=1) # (N, 3, 3)

    # sample points in unit cube
    whl = torch.norm(UVW, dim=2, keepdim=True)
    points = torch.rand((N, num_samples, 3), device=boxes.device) * whl[:, None, ..., 0]
    ones_vec = torch.ones(N, num_samples, 1).to(boxes.device)
    points = torch.cat([points, ones_vec], axis=-1)

    UVW = UVW / (whl + 1e-6) # (N, 3, 3)
    transformation_matrices = torch.zeros((N, 4, 4)).to(boxes.device)
    transformation_matrices[:, -1, -1] = 1.0
    transformation_matrices[:, :-1, -1] = boxes[:, 0]
    transformation_matrices[:, :-1, :-1] = UVW.permute(0, 2, 1)

    points = points @ transformation_matrices.permute(0, 2, 1)
    return points[..., :-1]


def sample_points_within_box_(box: torch.Tensor, num_samples: int = 10):
    """
    Sample points within a box defined by its 8 coordinates
    Args:
        box: tensor of shape (8, 3) of the box coordinates
        num_samples: int defining the number of samples
    Returns:
        points: (num_samples, 3) of points inside the box
    """
    assert box.shape[0] == 8 and box.shape[1] == 3
    xyzmin = box.min(0).values.view(1, 3)
    xyzmax = box.max(0).values.view(1, 3)


    # uniformly sample in a 3D unit cube
    uvw = torch.rand((num_samples, 3), device=box.device)

    points = uvw * (xyzmax - xyzmin) + xyzmin


    # because the box is not axis aligned we need to check wether
    # the points are within the box
    num_points = 0
    samples = []
    while num_points < num_samples:
        inside = is_point_inside_box(box, points)
        samples.append(points[inside].view(-1, 3))
        num_points += inside.sum()

    samples = torch.cat(samples, dim=0)
    return samples[1:num_samples]


# -------------------------------------------------- #
#          MAIN: BOX3D_OVERLAP_SAMPLING              #
# -------------------------------------------------- #


def box3d_overlap_sampling(
    box1: torch.Tensor, box2: torch.Tensor, num_samples: int = 50000, eps=1e-5
):
    """
    Computes the intersection of two boxes by sampling points
    """
    vol1 = box_volume(box1)
    vol2 = box_volume(box2)

    points1 = sample_points_within_boxes(box1[None], num_samples=num_samples)[0]
    points2 = sample_points_within_boxes(box2[None], num_samples=num_samples)[0]
    #
    isin21 = is_point_inside_box(box1, points2, eps)
    num21 = isin21.sum()
    isin12 = is_point_inside_box(box2, points1, eps)
    num12 = isin12.sum()
    #
    assert num12 <= num_samples
    assert num21 <= num_samples
    #
    inters = (vol1 * num12 + vol2 * num21) / 2.0
    union = vol1 * num_samples + vol2 * num_samples - inters
    return inters / union


def box3d_overlap_sampling_batched(
    box_query: torch.Tensor, boxes_target: torch.Tensor, num_samples: int = 5000, eps=1e-5):
    """
    Computes the intersection of two boxes by sampling points
    """
    vol_query = box_volume(box_query)
    vol_targets = box_volume(boxes_target)
    num_targets = vol_targets.shape[0]

    if box_query.shape[0] == 8:
        box_query = box_query[None]

    points_query = sample_points_within_boxes(box_query, num_samples=num_samples) # (1, num_samples, 3)
    points_targets = sample_points_within_boxes(boxes_target, num_samples=num_samples) # (N, num_samples, 3)

    points_targets = points_targets.view(-1, 3)
    isin21 = are_points_inside_boxes(box_query, points_targets, eps)
    isin21 = isin21.view(num_targets, num_samples)
    num21 = isin21.sum(-1)
    isin12 = are_points_inside_boxes(boxes_target, points_query, eps)
    num12 = isin12.sum(-1)

    assert torch.all(num12 <= num_samples)
    assert torch.all(num21 <= num_samples)

    # # # IoU = vol(A^B)/min(vol(A), vol(B))
    inters = (vol_query * num12 + vol_targets * num21) / 2.0
    inters = inters / num_samples
    min_volumes = torch.where(vol_targets > vol_query, vol_query, vol_targets)
    assert torch.all(inters / min_volumes >= 0), "got a negative iou"
    return inters, min_volumes
