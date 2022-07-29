#include <array>
#include <iostream>
#include <chrono>

//CGAL
#include <CGAL/Timer.h>
#include <CGAL/tags.h>
#include <CGAL/property_map.h>
#include <CGAL/IO/read_ply_points.h>
#include <CGAL/IO/write_ply_points.h>

#include <CGAL/compute_average_spacing.h>

#include <cxxopts.hpp>

#include <CGAL/pca_estimate_normals.h>
#include <CGAL/Shape_detection/Efficient_RANSAC.h>


#include <CGAL/Search_traits_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

#include "value_color.h"

#include <CGAL/Bbox_3.h>

#include <CGAL/optimal_bounding_box.h>
#include <CGAL/Surface_mesh.h>

//#include "include/json/json.h"
#include "include/pointclouds.h"
#include "include/cuboids.h"

#include <Eigen/SVD>

using namespace std;
using Eigen::MatrixXd;

// Type declarations.
typedef CGAL::Simple_cartesian<double>                       Kernel;
typedef Kernel::FT                                           FT;
typedef Kernel::Point_3                                      Point;
typedef Kernel::Vector_3                                     Vector;
typedef std::pair<Point, Kernel::Vector_3>                   Point_with_normal;
typedef std::vector<Point_with_normal>                       Pwn_vector;
typedef std::vector<Point>                                   Point_cloud;
typedef CGAL::Surface_mesh<Point>                            Surface_mesh;

typedef std::array<Point, 8>                                 Bbox;

typedef CGAL::Sequential_tag Concurrency_tag;
typedef CGAL::First_of_pair_property_map<Point_with_normal>  Point_map;
typedef CGAL::Second_of_pair_property_map<Point_with_normal> Normal_map;
typedef CGAL::Shape_detection::Efficient_RANSAC_traits
        <Kernel, Pwn_vector, Point_map, Normal_map>             Traits;
typedef CGAL::Shape_detection::Efficient_RANSAC<Traits> Efficient_ransac;

typedef CGAL::Shape_detection::Plane<Traits>            Plane;

typedef CGAL::Search_traits_3<Kernel> TreeTraits;
typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
typedef Neighbor_search::Tree kdTree;


struct Timeout_callback {
    mutable int nb;
    mutable CGAL::Timer timer;
    const double limit;
    Timeout_callback(double limit) :
            nb(0), limit(limit) {
        timer.start();
    }
    bool operator()(double advancement) const {
        // Avoid calling time() at every single iteration, which could
        // impact performances very badly.
        ++nb;
        if (nb % 1000 != 0)
            return true;
        // If the limit is reached, interrupt the algorithm.
        if (timer.time() > limit) {
            std::cerr << "Algorithm takes too long, exiting ("
                      << 100.0 * advancement << "% done)" << std::endl;
            return false;
        }
        return true;
    }
};


void write_pwn(const char *filename, const std::vector<Point_with_normal> &points) {
    std::ofstream ofile(filename, std::ios::binary);
    CGAL::set_binary_mode(ofile);
    CGAL::write_ply_points
            (ofile, points,
             CGAL::parameters::point_map(Point_map()).
                     normal_map(Normal_map()));
    ofile.close();
}

double squared_cosine_similarity(const Kernel::Vector_3 &u, const Kernel::Vector_3 &v) {
    if (u==CGAL::NULL_VECTOR || v==CGAL::NULL_VECTOR) {
        return 0;
    }
    FT u_dot_v = (u / CGAL::sqrt(u.squared_length())) * (v / CGAL::sqrt(v.squared_length()));
    return CGAL::to_double(u_dot_v);
}


bool check_contact_clusters(const Cluster &cluster_a,
                            const Cluster &cluster_b,
                            const double &distance_threshold,
                            const double &parallel_threshold,
                            const double &orthogonal_threshold) {
    return (minSquaredDistance(cluster_a.point_cloud, cluster_b.point_cloud) < distance_threshold * distance_threshold &&
            (squared_cosine_similarity(cluster_a.normal, cluster_b.normal) > parallel_threshold ||
             squared_cosine_similarity(cluster_a.normal, cluster_b.normal) < orthogonal_threshold));
}


int main (int argc, char** argv) {
    std::cout << "Efficient RANSAC" << std::endl;

    cxxopts::Options options(argv[0], "Computing sdf samples from points on camera rays");

    options.add_options()
            ("h,help", "Print usage")
            ("m,model_path", "Path to 3D scene PLY models", cxxopts::value<std::string>())
            ("o,outdir", "Output directory", cxxopts::value<std::string>()->default_value("."))
            ("p,miss_probability", "Probability to miss the largest primitive at each iteration.",
             cxxopts::value<double>()->default_value("0.05"))
            ("t,threshold", "Maximum Euclidean distance between a point and a shape.",
             cxxopts::value<double>()->default_value("0.02"))
            ("e,epsilon", "Maximum Euclidean distance between points to be clustered.",
             cxxopts::value<double>()->default_value("0.05"))
            ("k,num_points", "Minimal number of inlier points to detect shape",
             cxxopts::value<int>()->default_value("10"))
            ("c,num_neighbors", "Number of neighbors to compute normals",
             cxxopts::value<int>()->default_value("200"))
            ("n,normal_threshold", "Maximum normal deviation",cxxopts::value<double>()->default_value("0.7"))
            ("max_use_cluster", "Maximum number of cluster sampling", cxxopts::value<int>()->default_value("1"))
            ;

    auto args = options.parse(argc, argv);
    if (args.count("help"))
    {
        cout << options.help() << endl;
        exit(0);
    }
    string fname = args["model_path"].as<string>();
    string outdir = args["outdir"].as<string>();
    if (outdir[outdir.size()-1] != '/')
        outdir += '/';
    double probability = args["miss_probability"].as<double>();
    double threshold = args["threshold"].as<double>();
    double cluster_epsilon = args["epsilon"].as<double>();
    double normal_threshold = args["normal_threshold"].as<double>();
    int nb_neighbors_for_normals_computation = args["num_neighbors"].as<int>();
    int max_usage_per_cluster = args["max_use_cluster"].as<int>();

    int min_points = args["num_points"].as<int>();

    //// loading file with normals
    string outPath = outdir + "out_model.obj";
    ofstream outFile(outPath);

    std::vector<Point_with_normal> pwns;

    string pwnPath = outdir + "test.pwn";
    ifstream pwnFile(pwnPath, std::ios::binary);

    if (!pwnFile ||
        !CGAL::read_ply_points(pwnFile, std::back_inserter(pwns),
                               CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Point_with_normal>()).
                                       normal_map(Normal_map())))
    {
        pwnFile.close();
        // load ply file, compute normals
        std::ifstream input(fname);
        if (!input ||
            !CGAL::read_ply_points(input, std::back_inserter(pwns),
                                   CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Point_with_normal>()).
                                           normal_map(Normal_map())))
        {
            std::cerr << "Error: cannot read file " << fname << std::endl;
            return EXIT_FAILURE;
        }

        cout << "Computing normals ";
        // Estimates normals direction.
        // Note: pca_estimate_normals() requires a range of points
        // as well as property maps to access each point's position and normal.
        const int nb_neighbors = nb_neighbors_for_normals_computation; // K-nearest neighbors = 3 rings
        double spacing
                = CGAL::compute_average_spacing<Concurrency_tag>
                        (pwns, nb_neighbors,
                         CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Point_with_normal>()));

        // Then, estimate normals with a fixed radius
        CGAL::pca_estimate_normals<Concurrency_tag>
                (pwns,
                 0, // when using a neighborhood radius, K=0 means no limit on the number of neighbors returns
                 CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Point_with_normal>()).
                         normal_map(CGAL::Second_of_pair_property_map<Point_with_normal>()).
                         neighbor_radius(2. * spacing)); // use 2*spacing as neighborhood radius
        cout << "Done " << endl;

        input.close();
        write_pwn((outdir + "test.pwn").c_str(), pwns);
    }

    //// RANSAC part
    random_shuffle(colors.begin(), colors.end());

    cout << "Running RANSAC " << endl;
    // Set parameters for shape detection.
    Efficient_ransac::Parameters parameters;
    parameters.probability = probability;
    parameters.min_points = min_points;
    parameters.epsilon = threshold;
    parameters.cluster_epsilon = cluster_epsilon;
    parameters.normal_threshold = normal_threshold;

    Efficient_ransac ransac;
    ransac.set_input(pwns);
    ransac.add_shape_factory<Plane>();
    // Create callback that interrupts the algorithm if it takes too long.
    Timeout_callback timeout_callback(1000000);
    // Detect registered shapes with the default parameters.
    ransac.detect(parameters, timeout_callback);
    Efficient_ransac::Shape_range shapes = ransac.shapes();
    vector<Cluster> Clusters;
    int shape_count = 0;

    for (auto it = shapes.begin(); it != shapes.end(); it++)
    {
        Cluster cluster;
        int cluster_size = 0;
        Vector centroid(0.,0.,0.);

        if (Plane* plane = dynamic_cast<Plane*>(it->get())) {
            shape_count++;
            Kernel::Vector_3 normal = plane->plane_normal();
            cout << "Plane " << shape_count << ": " << plane->info() << endl;

            // Iterate through point indices assigned to each detected shape.
            // also add each point to a Point_set
            std::vector<std::size_t>::const_iterator index_it = (*it)->indices_of_assigned_points().begin();
            vector<int> color = colors[shape_count % (colors.size())];
            Point_cloud point_set;

            while (index_it != (*it)->indices_of_assigned_points().end()) {
                // Retrieve point.
                const Point_with_normal &p = *(pwns.begin() + (*index_it));
                point_set.push_back(p.first);

                outFile << "v " << CGAL::to_double(p.first.x()) << " "
                        << CGAL::to_double(p.first.y()) << " "
                        << CGAL::to_double(p.first.z()) << " "
                        << color[0] << " " << color[1] << " " << color[2] << endl;

                outFile << "vn " << CGAL::to_double(normal.x()) << " "
                        << CGAL::to_double(normal.y()) << " "
                        << CGAL::to_double(normal.z()) << endl;

                // Adds Euclidean distance between point and shape.
                // sum_distances += CGAL::sqrt((*it)->squared_distance(p.first));
                // Proceed with the next point.
                index_it++;
                cluster_size++;
                centroid += Vector(Point(0., 0., 0.), p.first);
            }
            cluster.point_cloud = point_set;
            cluster.normal = normal;
            cluster.centroid = Point(centroid.x()/cluster_size, centroid.y()/cluster_size, centroid.z()/cluster_size);
            Clusters.emplace_back(cluster);
        }
    }

    cout << "Found " << shape_count << " shapes!" << endl;

    cout << "Checking compatibility between clusters.";
    vector<vector<std::pair<int, int>>> compatible_clusters_map_with_count;
    vector<vector<int>> permutations;

    for (int i = 0; i!= shape_count; i++){
        compatible_clusters_map_with_count.emplace_back(vector<std::pair<int, int>>());
        permutations.emplace_back(vector<int>());
    }

    for (int i = 0; i != shape_count-1; i++) {
        int order=0;
#pragma omp parallel for
            for (int j = i + 1; j != shape_count; j++) {
#pragma omp critical
                {
                if (check_contact_clusters(Clusters[i], Clusters[j], cluster_epsilon, normal_threshold, 0.3)) {
                        compatible_clusters_map_with_count[i].push_back(std::pair<size_t, size_t>(j, max_usage_per_cluster));
                    }
                }
            }
    }
    int i,k,j;


    cout << "Done." << endl;

    size_t total_possible_bbox = 0;
    for (auto it=compatible_clusters_map_with_count.begin(); it!=compatible_clusters_map_with_count.end(); it++) {
        total_possible_bbox += (*it).size();
    }
    cout << "A total of " << 2*total_possible_bbox << " bounding boxes can be obtained." << endl;

    int n = 0;
    vector<Bbox> list_bboxes;
    vector<Bbox> list_shape_bboxes;
    Surface_mesh out_surfMesh;
    cout << "Computing boxes from pairs";

    for (int i=0; i!=shape_count; i++){
        if (!Clusters[i].point_cloud.empty()){
            std::array<Point, 8> shape_obb_points;
            CGAL::oriented_bounding_box(Clusters[i].point_cloud, shape_obb_points,
                                        CGAL::parameters::use_convex_hull(true));
            list_shape_bboxes.emplace_back(shape_obb_points);

            if (!compatible_clusters_map_with_count[i].empty()) {
                for (int j = 0; j != compatible_clusters_map_with_count[i].size(); ++j) {
                    Bbox tmp_box1, tmp_box2, output_box1, output_box2;
                    bbox_from_cluster_pair(tmp_box1, tmp_box2,
                                           Clusters[compatible_clusters_map_with_count[i][j].first], Clusters[i]);
                    CGAL::oriented_bounding_box(tmp_box1, output_box1, CGAL::parameters::use_convex_hull(false));
                    CGAL::oriented_bounding_box(tmp_box2, output_box2, CGAL::parameters::use_convex_hull(false));
                    list_bboxes.emplace_back(output_box1);
                    list_bboxes.emplace_back(output_box2);
                }
            }
        }
    }

    string prefix;
    prefix = outdir + "pair";
    write_output_files_from_bboxes(list_bboxes, prefix);
    prefix = outdir + "single";
    write_output_files_from_bboxes(list_shape_bboxes, prefix);

    cout << " Done." << endl;

    return EXIT_SUCCESS;
}
