//
// Created by michael on 17/08/2021.
//
//#pragma once

#ifndef PLANE_DETECTOR_POINTCLOUDS_H
#define PLANE_DETECTOR_POINTCLOUDS_H
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

typedef CGAL::Simple_cartesian<double>                       Kernel;
typedef Kernel::FT                                           FT;
typedef Kernel::Point_3                                      Point;
typedef Kernel::Vector_3                                     Vector;
typedef std::vector<Point>                                   Point_cloud;
typedef CGAL::Search_traits_3<Kernel> TreeTraits;
typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
typedef Neighbor_search::Tree kdTree;

struct Cluster {
    Point_cloud point_cloud;
    Point centroid;
    std::array<Point, 8> box_points;
    Vector normal;
};

double minSquaredDistance(const Point_cloud &A, const Point_cloud &B);
#endif //PLANE_DETECTOR_POINTCLOUDS_H
