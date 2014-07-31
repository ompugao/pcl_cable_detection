#include <iostream>
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/utility.hpp>
#include "pcl-cable-detection.h"
#include <pcl/io/ply_io.h>
#include <sstream>

using namespace pcl_cable_detection;

int main (int argc, char** argv)
{
    // command line arguments parsing /*{{{*/
    namespace bpo = boost::program_options;
    namespace bfs = boost::filesystem;
    bpo::options_description opts_desc("Allowed options");
    bpo::positional_options_description p;

    opts_desc.add_options()
        ("input_file", bpo::value< std::string >(), "Input data.")
        ("removeplane", bpo::value<bool>()->default_value(false), "findplane or not?")
        ("voxelsize", bpo::value< double >(), "voxelsize")
        ("distthreshold_findplane", bpo::value< double >(), "distance threshold for finding plane")
        ("cableradius", bpo::value< double >(), "cable radius")
        ("distthreshold_cylindermodel", bpo::value< double >(), "distance threshold for finding cylinder")
        ("scenesamplingradius", bpo::value< double >(), "scene sampling radius")
        ("cableterminalply", bpo::value< std::string >(), "path to terminal ply file")
    ;

    bpo::variables_map opts;
    bpo::store(bpo::parse_command_line(argc, argv, opts_desc, bpo::command_line_style::unix_style ^ bpo::command_line_style::allow_short), opts);
    bool badargs = false;
    try { bpo::notify(opts); }
    catch(...) { badargs = true; }
    if(opts.count("help") || badargs) {
        //std::cout << "Usage: " << bfs::basename(argv[0]) << " INPUT_DIR OUTPUT_DIR [OPTS]" << std::endl;
        //std::cout << std::endl;
        std::cout << opts_desc << std::endl;
        return (1);
    }

    std::string input_file = opts["input_file"].as<std::string> ();

    double voxelsize = opts["voxelsize"].as<double>();
    bool removeplane= opts["removeplane"].as<bool>();
    double distthreshold_findplane= opts["distthreshold_findplane"].as<double>();
    double cableradius = opts["cableradius"].as<double>();
    double distthreshold_cylindermodel= opts["distthreshold_cylindermodel"].as<double>();
    double scenesamplingradius = opts["scenesamplingradius"].as<double>();
    std::string cableterminalply = opts["cableterminalply"].as<std::string>();
/*}}}*/
    std::cout << "loading pcd file..." << std::endl; /*{{{*/
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointNormal>);
    pcl::io::loadPCDFile (input_file, *cloud);
/*}}}*/
    if(removeplane) { /*{{{*/
        std::cout << "finding plane..." << std::endl;
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        if (!pcl_cable_detection::findPlane<pcl::PointNormal>(cloud, voxelsize, distthreshold_findplane, *inliers, *coefficients)) {
            PCL_ERROR("could not find plane.");
        }

        std::cerr << "Model coefficients: " << coefficients->values[0] << " "
                  << coefficients->values[1] << " "
                  << coefficients->values[2] << " "
                  << coefficients->values[3] << std::endl;

        // Remove the planar inliers, extract the rest
        std::cout << "remove plane..." << std::endl;
        pcl::ExtractIndices<pcl::PointNormal> extract;
        extract.setInputCloud (cloud);
        extract.setIndices (inliers);
        extract.setNegative (true);
        extract.filter (*cloud_filtered);
        //pcl::copyPointCloud (*cloud, *cloud_filtered);
    } else {
        pcl::copyPointCloud (*cloud, *cloud_filtered);
    }
/*}}}*/
    std::cout << "compute curvature..." << std::endl; /*{{{*/
    pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> ne;
    std::cout << "the number of points: " << cloud->points.size() << std::endl;
    std::cout << "width: " << cloud->width << " height: " << cloud->height << std::endl;
    ne.setInputCloud (cloud_filtered);
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointNormal> ());
    ne.setSearchMethod (tree);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud (*cloud_filtered, *cloud_normals); //copy xyz
    ne.setRadiusSearch (distthreshold_cylindermodel);
    ne.compute (*cloud_normals);
    for (size_t i = 0; i < cloud_normals->points.size(); i++) {
        if (pcl_isnan(cloud_normals->points[i].curvature)) {
            cloud_normals->points[i].curvature = 0.0;
        }
        // revert original normals
        //cloud_normals->points[i].normal_x = cloud_filtered->points[i].normal_x;
        //cloud_normals->points[i].normal_y = cloud_filtered->points[i].normal_y;
        //cloud_normals->points[i].normal_z = cloud_filtered->points[i].normal_z;
        //std::cout << cloud_normals->points[i].curvature << std::endl;
    }
    std::cout << "finished computing normals! size: " << cloud_normals->size() << std::endl;
/*}}}*/

    pcl::PointCloud<pcl::PointNormal>::Ptr terminalcloud(new pcl::PointCloud<pcl::PointNormal>());
    pcl::io::loadPLYFile(cableterminalply, *terminalcloud);
    Eigen::Vector3f axis(0,-1,0);

    CableDetection<pcl::PointNormal> cabledetection(terminalcloud, axis);

    cabledetection.setCableRadius(cableradius);
    cabledetection.setThresholdCylinderModel(distthreshold_cylindermodel);
    cabledetection.setSceneSamplingRadius(scenesamplingradius);
    cabledetection.setInputCloud(cloud_normals);
    cabledetection.RunViewer();
    //pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colorcloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);/*{{{*/
    //pcl::copyPointCloud (*cloud_normals, *colorcloud);
    //for (size_t i = 0; i < colorcloud->size(); i++) {
    //colorcloud->points[i].r = 255.0;
    //colorcloud->points[i].g = 255.0;
    //colorcloud->points[i].b = 0;
    //}
    /*************/
    //pcl::visualization::CloudViewer viewer ("cloud viewer");
    //viewer.showCloud (colorcloud);
    //while (!viewer.wasStopped ())
    //{
    //}/*}}}*/
    return (0);
}
