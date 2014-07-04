#include <iostream>
#include <boost/program_options.hpp>
#include <pcl/io/pcd_io.h>
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

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile (input_file, *cloud);

    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    if (!pcl_cable_detection::findPlane<pcl::PointXYZ>(cloud, *inliers, *coefficients, voxelsize, threshold)) {
        PCL_ERROR("could not find plane.");
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud (*cloud, *colorcloud);

    for (size_t i = 0; i < colorcloud->size(); i++) {
        colorcloud->points[i].r = 255;
        colorcloud->points[i].g = 255;
        colorcloud->points[i].b = 255;
    }

    for (size_t i = 0; i < inliers->indices.size (); ++i) {
        colorcloud->points[inliers->indices[i]].r = 255;
        colorcloud->points[inliers->indices[i]].g = 0;
        colorcloud->points[inliers->indices[i]].b = 0;
    }


    std::cerr << "Model coefficients: " << coefficients->values[0] << " "
              << coefficients->values[1] << " "
              << coefficients->values[2] << " "
              << coefficients->values[3] << std::endl;

    std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
    pcl::visualization::CloudViewer viewer ("cloud viewer");
    viewer.showCloud (colorcloud);
    while (!viewer.wasStopped ())
    {
    }

    return (0);
}
