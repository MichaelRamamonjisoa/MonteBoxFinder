//
// Created by michael on 17/08/2021.
//
//#pragma once

#ifndef PLANE_DETECTOR_CUBOIDS_H
#define PLANE_DETECTOR_CUBOIDS_H
#include <CGAL/optimal_bounding_box.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/distance.h>

#include "pointclouds.h"
#include <json/json.h>
#include <json/json-forwards.h>

typedef CGAL::Simple_cartesian<double>                       Kernel;
typedef Kernel::FT                                           FT;
typedef Kernel::Point_3                                      Point;
typedef Kernel::Vector_3                                     Vector;
typedef std::array<Point, 8>                                 Bbox;
typedef std::vector<Point>                                   Point_cloud;
//typedef std::pair<Point_cloud, Kernel::Vector_3>             Cluster;

typedef CGAL::Surface_mesh<Point>                            Surface_mesh;
typedef Surface_mesh::Vertex_index                           vertex_descriptor;
typedef Surface_mesh::Face_index                             face_descriptor;


std::vector<std::pair<Point, Point>> get_lines_from_bbox(std::array<Point, 8> &bbox_points);
void write_output_files_from_bboxes(std::vector<Bbox> &listBboxes, std::string &filePrefix);
void bbox_from_cluster_pair(Bbox &output_box1, Bbox &output_box2, const Cluster &A, const Cluster &B);
void gram_schmidt(const Vector &a, const Vector &b, std::array<Vector, 3> &uvw);
void bbox_from_axes_and_projections(Bbox &outbox, const std::array<Vector, 3> &uvw, const std::array<FT, 3> &min_proj, const std::array<FT, 3> &max_proj);
void bbox_from_point_axes_and_lengths(Bbox &outbox, const std::array<Vector, 3> &uvw, const std::array<FT, 3> &LWH, const Point &m);
void bbox_from_point_axes_lengths_and_center(Bbox &outbox, const std::array<Vector, 3> &uvw, const std::array<FT, 3> &LWH, const Point &m);

bool test_intersections();

#endif //PLANE_DETECTOR_CUBOIDS_H