//
// Created by michael on 17/08/2021.
//

#include "../include/pointclouds.h"
#include "../include/cuboids.h"

double minSquaredDistance(const Point_cloud &A, const Point_cloud &B)
{
    FT bestSquaredDistance = DBL_MAX;
    Point_cloud query_pc;
    kdTree tree;

    if (B.size() > A.size()){
        query_pc = Point_cloud(A.begin(), A.end());
        for (auto pt=B.begin(); pt!=B.end(); pt++){
            tree.insert(*pt);
        }
    } else {
        query_pc = Point_cloud(B.begin(), B.end());
        for (auto pt=A.begin(); pt!=A.end(); pt++){
            tree.insert(*pt);
        }
    }

#pragma omp parallel for
    for(const auto& point: query_pc)
    {
        Neighbor_search search(tree, point, 1);
        FT curDistance = search.begin()->second;
#pragma omp critical
        {
            if(curDistance < bestSquaredDistance)
                bestSquaredDistance = curDistance;
        }
    }
    return CGAL::to_double(bestSquaredDistance);
}
