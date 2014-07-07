#include <iostream>
#include <boost/program_options.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>
#include "pcl-cable-detection.h"

int main (int argc, char** argv)
{
    namespace bpo = boost::program_options;
    namespace bfs = boost::filesystem;
    bpo::options_description opts_desc("Allowed options");
    bpo::positional_options_description p;

    opts_desc.add_options()
        ("input_file", bpo::value< std::string >(), "Input data.")
        ("voxelsize", bpo::value< double >(), "voxelsize")
        ("threshold", bpo::value< double >(), "distance threshold")
    ;

    bpo::variables_map opts;
    bpo::store(bpo::parse_command_line(argc, argv, opts_desc, bpo::command_line_style::unix_style ^ bpo::command_line_style::allow_short), opts);
    bool badargs = false;
    try { bpo::notify(opts); }
    catch(...) { badargs = true; }
    if(opts.count("help") || badargs) {
        std::cout << "Usage: " << bfs::basename(argv[0]) << " INPUT_DIR OUTPUT_DIR [OPTS]" << std::endl;
        std::cout << std::endl;
        std::cout << opts_desc << std::endl;
        return (1);
    }

    std::string input_file = opts["input_file"].as<std::string> ();
    double voxelsize = opts["voxelsize"].as<double>();
    double threshold = opts["threshold"].as<double>();

    //PCL_INFO("loading pcd file...");
    std::cout << "loading pcd file..." << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile (input_file, *cloud);

    //PCL_INFO("finding plane...");
    std::cout << "finding plane..." << std::endl;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    if (!pcl_cable_detection::findPlane<pcl::PointXYZ>(cloud, *inliers, *coefficients, voxelsize, threshold)) {
        PCL_ERROR("could not find plane.");
    }

    std::cerr << "Model coefficients: " << coefficients->values[0] << " "
              << coefficients->values[1] << " "
              << coefficients->values[2] << " "
              << coefficients->values[3] << std::endl;

    // Remove the planar inliers, extract the rest
    std::cout << "remove plane..." << std::endl;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (true);
    extract.filter (*cloud_filtered);
    //extract_normals.setNegative (true);
    //extract_normals.setInputCloud (cloud_normals);
    //extract_normals.setIndices (inliers_plane);
    //extract_normals.filter (*cloud_normals2);

    //std::cout << "Create voxel grid" << std::endl;
    //pcl::VoxelGrid<pcl::PointXYZ> vg;
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered2 (new pcl::PointCloud<pcl::PointXYZ>);
    //vg.setInputCloud (cloud_filtered);
    //vg.setLeafSize (0.0005, 0.0005, 0.0005);
    //vg.filter (*cloud_filtered2);
    //std::cout << cloud_filtered2->points.size() << std::endl;

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (cloud_filtered);
    std::cout << "computing normals..." << std::endl;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch (0.002);
    ne.compute (*cloud_normals);
    std::cout << "finished computing normals!" << std::endl;

    // visualize normals
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud (*cloud_filtered, *colorcloud);

    for (size_t i = 0; i < colorcloud->size(); i++) {
        colorcloud->points[i].r = 255;
        colorcloud->points[i].g = 0;
        colorcloud->points[i].b = 0;
    }

    //viewer.setBackgroundColor (0.0, 0.0, 0.0);
    viewer.addPointCloudNormals<pcl::PointXYZRGB,pcl::Normal>(colorcloud, cloud_normals, 100, 0.01f);
    while (!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }

    //pcl::visualization::CloudViewer viewer ("cloud viewer");
    //viewer.showCloud (colorcloud);
    //while (!viewer.wasStopped ())
    //{
    //}
    return (0);
}
