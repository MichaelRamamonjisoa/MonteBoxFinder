import open3d as o3d
import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt
import imageio
from PIL import Image, ImageDraw, ImageFont
import lib.visualization_callbacks as callbacks


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {list} -- list of radii of cylinder, or single radius (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = np.array(radius)
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            radius = self.radius if len(self.radius) == 1 else self.radius[i]
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                radius, line_length)
            cylinder_segment.simplify_quadric_decimation(15)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=cylinder_segment.get_center())
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def bbox_to_lineset(box, color=[0, 1, 0]):
    points = box
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [1, 6], [6, 7], [2, 7], [5, 6],
         [4, 5], [4, 7], [0, 5], [3, 4]]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def bboxes_to_lineset(boxes, color=[0, 1, 0]):
    points = []
    lines = []
    boxlines = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [1, 6], [6, 7], [2, 7], [5, 6],
                         [4, 5], [4, 7], [0, 5], [3, 4]])
    for box in boxes:
        points.extend(box)
        lines += boxlines.tolist()
        boxlines += 8

    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def bbox_to_thickbox(box, thickness=0.05, color=[0, 1, 0]):
    #     using CGAL convention
    points = box
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [1, 6], [6, 7], [2, 7], [5, 6],
         [4, 5], [4, 7], [0, 5], [3, 4]]
    colors = [color for _ in range(len(lines))]
    line_mesh = LineMesh(points, lines, colors, radius=thickness)
    return line_mesh.cylinder_segments


def merge_meshes(meshes, color=[1, 0, 0]):
    full_mesh = o3d.geometry.TriangleMesh()
    vertices = []
    triangles = []
    vert_idx = 0
    for mesh in meshes:
        vertices.extend(mesh.vertices)
        triangles.extend((np.array(mesh.triangles) + vert_idx).tolist())
        vert_idx += len(mesh.vertices)
    full_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    full_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    full_mesh.paint_uniform_color(color)
    return full_mesh

import time

def bboxes_to_thickboxes(boxes, thickness=0.05, color=[0, 1, 0]):
    points = []
    lines = []
    colors = []
    thicknesses = []
    boxlines = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [1, 6], [6, 7], [2, 7], [5, 6],
                         [4, 5], [4, 7], [0, 5], [3, 4]])
    if len(boxes)==0:
        return None

    for i, box in enumerate(boxes):
        points.extend(box)
        lines += boxlines.tolist()
        boxlines += 8
        colors += [color for _ in range(12)]
        if isinstance(thickness, float):
            thicknesses += [thickness for _ in range(12)]
        else:
            thicknesses += [thickness[i] for _ in range(12)]
    line_mesh = LineMesh(points, lines, colors, radius=thicknesses)
    return merge_meshes(line_mesh.cylinder_segments, color)


def bboxes_to_thickboxes_p3d(boxes, thickness=0.05, color=[0, 1, 0]):
    points = []
    lines = []
    boxlines = np.array([[0, 1], [1, 2], [2, 3], [1, 5], [5, 6], [2, 6], [4, 5], [4, 7],
                         [6, 7], [0, 4], [0, 3], [3, 7]])
    if len(boxes)==0:
        return None

    for box in boxes:
        points.extend(box)
        lines += boxlines.tolist()
        boxlines += 8

    colors = [color for i in range(len(lines))]
    line_mesh = LineMesh(points, lines, colors, radius=thickness)
    return merge_meshes(line_mesh.cylinder_segments, color)


def draw_line(pt1, pt2, color=[1,0,0], thickness=0.05):
    line = [[0,1]]
    line_mesh = LineMesh([pt1, pt2], [[0,1]], [color], radius=thickness)
    return line_mesh.cylinder_segments


def custom_draw_geometry_with_rotation(pcd):
    """
    From Open3D doc http://www.open3d.org/docs/0.9.0/tutorial/Advanced/customized_visualization.html
    """

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              callbacks.rotate_view)


def draw_and_rotate_prediction(pcd, prediction_path, angle_increment=10., boxes_color=[0, 1, 0],
                               custom_text="",
                               outdir=""):
    try:
        with open(prediction_path, "r") as f:
            predictions = json.load(f)
            score = predictions["score"]
            predictions = predictions["bbox"]
        # compute geometry for bounding boxes
        if len(predictions) > 0:
            bboxes = bboxes_to_thickboxes([box_pts for box_pts in predictions],
                                          thickness=0.02, color=boxes_color)
    except FileNotFoundError as e:
        bboxes = None

    # the function can have attributes, in this case an Open3D visualizer
    draw_and_rotate_prediction.vis = o3d.visualization.Visualizer()
    draw_and_rotate_prediction.index = 0
    draw_and_rotate_prediction.angle = 0
    draw_and_rotate_prediction.angle_increment = angle_increment
    draw_and_rotate_prediction.custom_text = custom_text
    draw_and_rotate_prediction.score = score

    outdir = os.path.join(outdir, "image")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    draw_and_rotate_prediction.outdir = outdir

    def rotate_and_save(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. (Re-render)
        ctr = vis.get_view_control()
        glb = draw_and_rotate_prediction
        if glb.angle < 1000:
            image = vis.capture_screen_float_buffer(False)
            pil_image = np.asarray(image)
            _, screen_width, _ = pil_image.shape
            pil_image = Image.fromarray(np.uint8(pil_image*255))
            drawer = ImageDraw.Draw(pil_image)
            # Add Text to an image
            drawer.text((10, 10), glb.custom_text + "\n Score: {:.04f}".format(glb.score),
                        font=ImageFont.truetype('FreeMono.ttf', 65),
                        fill=(0, 0, 0))
            pil_image.save(os.path.join(draw_and_rotate_prediction.outdir, "{:05d}.jpg".format(glb.index)))
            # plt.imsave(os.path.join(draw_and_rotate_prediction.outdir, "{:05d}.jpg".format(glb.index)),
            #                         np.asarray(image), dpi = 1)
            #vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            #vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.angle = glb.angle + glb.angle_increment
        glb.index = glb.index + 1
        ctr.rotate(angle_increment, 0.0)
        if glb.index < 360:
            pass
        else:
            # remove callback
            draw_and_rotate_prediction.vis.register_animation_callback(None)
        return False

    vis = draw_and_rotate_prediction.vis
    vis.create_window()
    vis.add_geometry(pcd)
    if bboxes is not None:
        vis.add_geometry(bboxes)
    # vis.get_render_option().load_from_json("../../TestData/renderoption.json")

    # render result with rotation + saving callback
    vis.register_animation_callback(rotate_and_save)
    vis.run()
    vis.destroy_window()
    vis.close()

    # create GIF out of images
    print("Writing GIF")
    filenames = [os.path.join(draw_and_rotate_prediction.outdir, file)
                 for file in os.listdir(os.path.join(draw_and_rotate_prediction.outdir))
                 if file.endswith(".jpg")]
    filenames = sorted(filenames)
    prediction_path = prediction_path.replace(".json", ".gif")
    with imageio.get_writer(prediction_path, mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count