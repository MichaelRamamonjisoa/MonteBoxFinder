//
// Created by michael on 21/07/2021.
//

#include "../include/cuboids.h"

#include <fstream>

using namespace std;
namespace PMP = CGAL::Polygon_mesh_processing;

vector<pair<Point, Point>> get_lines_from_bbox(array<Point, 8> &bbox_points){
    vector<pair<Point, Point>> output_list;
    output_list.push_back(pair<Point, Point> (bbox_points[0], bbox_points[1]));
    output_list.push_back(pair<Point, Point> (bbox_points[1], bbox_points[2]));
    output_list.push_back(pair<Point, Point> (bbox_points[2], bbox_points[3]));
    output_list.push_back(pair<Point, Point> (bbox_points[3], bbox_points[0]));
    output_list.push_back(pair<Point, Point> (bbox_points[0], bbox_points[5]));
    output_list.push_back(pair<Point, Point> (bbox_points[5], bbox_points[4]));
    output_list.push_back(pair<Point, Point> (bbox_points[4], bbox_points[3]));
    output_list.push_back(pair<Point, Point> (bbox_points[4], bbox_points[7]));
    output_list.push_back(pair<Point, Point> (bbox_points[7], bbox_points[6]));
    output_list.push_back(pair<Point, Point> (bbox_points[6], bbox_points[5]));
    output_list.push_back(pair<Point, Point> (bbox_points[6], bbox_points[1]));
    output_list.push_back(pair<Point, Point> (bbox_points[7], bbox_points[2]));

    return output_list;
}

void write_output_files_from_bboxes(vector<Bbox> &listBboxes, string &filePrefix){
    // Saving bounding boxes as [[pt[i][dim] for dim in range(3)] for i in range(8)]
    Surface_mesh out_surfMesh;

    ofstream bboxesFile;
    string bboxesPath = filePrefix + "_bboxes.json";
    bboxesFile.open(bboxesPath);

    string outLinesPath = filePrefix + "_lines.ply";
    ofstream outLinesFile(outLinesPath);

    Json::StyledWriter styledWriter;
    Json::Value boxes_root;
    Json::Value json_boxes(Json::arrayValue);

    for (auto obb_points : listBboxes) {
        Json::Value bbox(Json::arrayValue);
        for (auto point : obb_points){
            Json::Value pt_root(Json::arrayValue);
            pt_root.append(point.x());
            pt_root.append(point.y());
            pt_root.append(point.z());
            bbox.append(pt_root);
        }
        json_boxes.append(bbox);

        vector<pair<Point, Point>> list_lines = get_lines_from_bbox(obb_points);
        for (auto line : list_lines){
            vertex_descriptor u = out_surfMesh.add_vertex(line.first);
            vertex_descriptor v = out_surfMesh.add_vertex(line.second);
            vertex_descriptor w = out_surfMesh.add_vertex(line.first);
            out_surfMesh.add_face(u,v,w);
        }
    }

    boxes_root["bbox"] = json_boxes;
    bboxesFile << styledWriter.write(boxes_root);
    bboxesFile.close();

    CGAL::write_ply(outLinesFile, out_surfMesh);
    outLinesFile.close();
}


void gram_schmidt(const Vector &a, const Vector &b, std::array<Vector, 3> &uvw){
    FT norm_a = CGAL::sqrt(a.squared_length());
    Vector u = a / norm_a;
    Vector v = b - CGAL::scalar_product(u, b) * u;
    v = v / CGAL::sqrt(v.squared_length());
    Vector w = CGAL::cross_product(a, b);
    w = w / CGAL::sqrt(w.squared_length());
    uvw = {u, v, w};
}


void bbox_from_point_axes_and_lengths(Bbox &outbox, const std::array<Vector, 3> &uvw, const std::array<FT, 3> &LWH, const Point &m){
    Point p0(m.x(), m.y(), m.z());
    Point p1, p2, p3, p4, p5, p6, p7;
    Vector U(uvw[0].x(), uvw[0].y(), uvw[0].z());
    Vector V(uvw[1].x(), uvw[1].y(), uvw[1].z());
    Vector W(uvw[2].x(), uvw[2].y(), uvw[2].z());

    U *= LWH[0];
    V *= LWH[1];
    W *= LWH[2];


    p1 = p0 + U;
    p2 = p0 + U + V;
    p3 = p0 +     V;
    p4 = p0 +     V + W;
    p5 = p0 +         W;
    p6 = p0 + U     + W;
    p7 = p0 + U + V + W;

    outbox = {p0, p1, p2, p3, p4, p5, p6, p7};
}

void bbox_from_point_axes_lengths_and_center(Bbox &outbox, const std::array<Vector, 3> &uvw, const std::array<FT, 3> &LWH, const Point &m){
    Vector U(uvw[0].x(), uvw[0].y(), uvw[0].z());
    Vector V(uvw[1].x(), uvw[1].y(), uvw[1].z());
    Vector W(uvw[2].x(), uvw[2].y(), uvw[2].z());

    Point p0, p1, p2, p3, p4, p5, p6, p7;

    U *= LWH[0];
    V *= LWH[1];
    W *= LWH[2];

    p0 = m - 0.5 * U - 0.5 * V - 0.5 * W;
    p1 = p0 + U;
    p2 = p0 + U + V;
    p3 = p0 +     V;
    p4 = p0 +     V + W;
    p5 = p0 +         W;
    p6 = p0 + U     + W;
    p7 = p0 + U + V + W;

    outbox = {p0, p1, p2, p3, p4, p5, p6, p7};
}


void bbox_from_axes_and_projections(Bbox &outbox, const std::array<Vector, 3> &uvw, const std::array<FT, 3> &min_proj, const std::array<FT, 3> &max_proj){
    FT uxm = uvw[0].x() * min_proj[0];
    FT uym = uvw[0].y() * min_proj[0];
    FT uzm = uvw[0].z() * min_proj[0];
    FT uxM = uvw[0].x() * max_proj[0];
    FT uyM = uvw[0].y() * max_proj[0];
    FT uzM = uvw[0].z() * max_proj[0];

    FT vxm = uvw[1].x() * min_proj[1];
    FT vym = uvw[1].y() * min_proj[1];
    FT vzm = uvw[1].z() * min_proj[1];
    FT vxM = uvw[1].x() * max_proj[1];
    FT vyM = uvw[1].y() * max_proj[1];
    FT vzM = uvw[1].z() * max_proj[1];

    FT wxm = uvw[2].x() * min_proj[2];
    FT wym = uvw[2].y() * min_proj[2];
    FT wzm = uvw[2].z() * min_proj[2];
    FT wxM = uvw[2].x() * max_proj[2];
    FT wyM = uvw[2].y() * max_proj[2];
    FT wzM = uvw[2].z() * max_proj[2];

    Point p0(uxm + vxm + wxm, uym + vym + wym, uzm + vzm + wzm);
    Point p1(uxM + vxm + wxm, uyM + vym + wym, uzM + vzm + wzm);
    Point p2(uxM + vxM + wxm, uyM + vyM + wym, uzM + vzM + wzm);
    Point p3(uxm + vxM + wxm, uym + vyM + wym, uzm + vzM + wzm);
    Point p4(uxm + vxM + wxM, uym + vyM + wyM, uzm + vzM + wzM);
    Point p5(uxm + vxm + wxM, uym + vym + wyM, uzm + vzm + wzM);
    Point p6(uxM + vxm + wxM, uyM + vym + wyM, uzM + vzm + wzM);
    Point p7(uxM + vxM + wxM, uyM + vyM + wyM, uzM + vzM + wzM);

    outbox = {p0, p1, p2, p3, p4, p5, p6, p7};
}


void bbox_from_cluster_pair(Bbox &output_box1, Bbox &output_box2, const Cluster &A, const Cluster &B){
    /**
     * Compute two versions of bounding boxes from a pair of quasi-orthogonal clusters
     */


    std::array<Vector, 3> uvw1, uvw2;
    gram_schmidt(A.normal, B.normal, uvw1);
    gram_schmidt(B.normal, A.normal, uvw2);

    std::array<FT, 3> min_proj_uvw1, max_proj_uvw1, min_proj_uvw2, max_proj_uvw2;
    std::array<FT, 3> LWH_1, LWH_2;

    FT min_proj = DBL_MAX;
    FT max_proj = -DBL_MAX;
    FT proj;

    for (int j=0; j<uvw1.size(); j++){

        min_proj = DBL_MAX;
        max_proj = -DBL_MAX;
        for (auto pt=A.point_cloud.begin(); pt!=A.point_cloud.end(); pt++){
            proj = (Vector(pt->x(), pt->y(), pt->z()) * uvw1[j]);
            if (proj > max_proj) max_proj = proj;
            if (proj < min_proj) min_proj = proj;
        }
        for (auto pt=B.point_cloud.begin(); pt!=B.point_cloud.end(); pt++){
            proj = (Vector(pt->x(), pt->y(), pt->z()) * uvw1[j]);
            if (proj > max_proj) max_proj = proj;
            if (proj < min_proj) min_proj = proj;
        }
        min_proj_uvw1[j] = min_proj;
        max_proj_uvw1[j] = max_proj;
        LWH_1[j] = (max_proj - min_proj);
    }

    Point m1(min_proj_uvw1[0] * uvw1[0].x() + min_proj_uvw1[1] * uvw1[1].x() + min_proj_uvw1[2] * uvw1[2].x(),
             min_proj_uvw1[0] * uvw1[0].y() + min_proj_uvw1[1] * uvw1[1].y() + min_proj_uvw1[2] * uvw1[2].y(),
             min_proj_uvw1[0] * uvw1[0].z() + min_proj_uvw1[1] * uvw1[1].z() + min_proj_uvw1[2] * uvw1[2].z());

    bbox_from_point_axes_and_lengths(output_box1, uvw1, LWH_1, m1);

    for (int j=0; j!=uvw2.size(); j++){
        min_proj = DBL_MAX;
        max_proj = -DBL_MAX;
        for (auto pt=A.point_cloud.begin(); pt!=A.point_cloud.end(); pt++){
            proj = (Vector(pt->x(), pt->y(), pt->z()) * uvw2[j]);
            if (proj > max_proj) max_proj = proj;
            if (proj < min_proj) min_proj = proj;
        }
        for (auto pt=B.point_cloud.begin(); pt!=B.point_cloud.end(); pt++){
            proj = (Vector(pt->x(), pt->y(), pt->z()) * uvw2[j]);
            if (proj > max_proj) max_proj = proj;
            if (proj < min_proj) min_proj = proj;
        }

        min_proj_uvw2[j] = min_proj;
        max_proj_uvw2[j] = max_proj;
        LWH_2[j] = (max_proj - min_proj);
    }

    Point m2(min_proj_uvw2[0] * uvw2[0].x() + min_proj_uvw2[1] * uvw2[1].x() + min_proj_uvw2[2] * uvw2[2].x(),
             min_proj_uvw2[0] * uvw2[0].y() + min_proj_uvw2[1] * uvw2[1].y() + min_proj_uvw2[2] * uvw2[2].y(),
             min_proj_uvw2[0] * uvw2[0].z() + min_proj_uvw2[1] * uvw2[1].z() + min_proj_uvw2[2] * uvw2[2].z());

    bbox_from_point_axes_and_lengths(output_box2, uvw2, LWH_2, m2);
}