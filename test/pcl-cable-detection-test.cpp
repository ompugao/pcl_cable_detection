#include <iostream>
#include <boost/program_options.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>
#include "pcl-cable-detection.h"

class CableDetection {
public:
    CableDetection(typename pcl::PointCloud<pcl::PointNormal>::Ptr input, double searchradius, double distthreshold_cylindermodel)
    {
        input_ = input;
        // Set up the full indices set
        std::vector<int> full_indices (input_->points.size ());
        indices_.reset(new pcl::PointIndices());
        indices_->indices.resize(input_->points.size());
        for (int fii = 0; fii < static_cast<int> (indices_->indices.size ()); ++fii) {  // fii = full indices iterator
            indices_->indices[fii] = fii;
        }
        searchradius_ = searchradius;
        distthreshold_cylindermodel_ = distthreshold_cylindermodel;
        viewer_.reset(new pcl::visualization::PCLVisualizer("PCL viewer_"));
        viewer_->setBackgroundColor (0.0, 0.0, 0.0);
        viewer_->addPointCloud<pcl::PointNormal> (input_, "cloud");
        viewer_->registerAreaPickingCallback(boost::bind(&CableDetection::area_picking_callback, this,_1));
        viewer_->registerPointPickingCallback(boost::bind(&CableDetection::point_picking_callback, this, _1));
        while (!viewer_->wasStopped ())
        {
            viewer_->spinOnce ();
        }
    }

    void point_picking_callback(const pcl::visualization::PointPickingEvent& event)
    {
        pcl::PointXYZ pt;
        event.getPoint (pt.x, pt.y, pt.z);
        size_t idx = event.getPointIndex ();
        std::cout << "picking point index: " << idx << std::endl;

        // get the points close to the picked point
        //std::vector<int> k_indices;
        pcl::PointIndices::Ptr k_indices(new pcl::PointIndices());
        std::vector<float> k_sqr_distances;
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointNormal>());
        tree->setInputCloud(input_);
        tree->radiusSearch (input_->points[idx], searchradius_, k_indices->indices, k_sqr_distances);

        pcl::ExtractIndices<pcl::PointNormal> extract;
        pcl::PointCloud<pcl::PointNormal>::Ptr closepoints(new pcl::PointCloud<pcl::PointNormal>());
        extract.setInputCloud (input_);
        extract.setIndices (k_indices);
        //extract.setNegative (true);
        extract.filter (*closepoints);


        // Create the segmentation object
        pcl::SACSegmentationFromNormals<pcl::PointNormal, pcl::PointNormal> seg;
        pcl::PointIndices::Ptr cylinderinlierindices(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr cylindercoeffs (new pcl::ModelCoefficients);
        // Optional
        seg.setOptimizeCoefficients (true);
        //seg.setAxis(...);
        // Mandatory
        seg.setModelType (pcl::SACMODEL_CYLINDER);
        seg.setMethodType (pcl::SAC_RANSAC);
        //seg.setMethodType (pcl::SAC_RRANSAC);
        seg.setMaxIterations(10000);
        seg.setDistanceThreshold (distthreshold_cylindermodel_);
        seg.setInputCloud (closepoints);
        seg.setInputNormals (closepoints);
        seg.segment (*cylinderinlierindices, *cylindercoeffs);

        std::cerr << "cylinder Model cylindercoeffs: " << cylindercoeffs->values[0] << " "
                  << cylindercoeffs->values[1] << " "
                  << cylindercoeffs->values[2] << " "
                  << cylindercoeffs->values[3] << " "
                  << cylindercoeffs->values[4] << " "
                  << cylindercoeffs->values[5] << " "
                  << cylindercoeffs->values[6] << " " << std::endl;

        viewer_->removeShape("cylinder");
        viewer_->addCylinder(*cylindercoeffs);

        std::cout << "extract model" << std::endl;
        pcl::PointCloud<pcl::PointNormal>::Ptr extractedpoints(new pcl::PointCloud<pcl::PointNormal>());
        extract.setInputCloud (closepoints);
        extract.setIndices (cylinderinlierindices);
        //extract.setNegative (true);
        extract.filter (*extractedpoints);
        if (extractedpoints->points.size() > 0) {
            pcl::io::savePCDFileBinaryCompressed ("extractedpoints.pcd", *extractedpoints);
        }
    }
    void area_picking_callback (const pcl::visualization::AreaPickingEvent &event)/*{{{*/
    {
        if (event.getPointsIndices (indices_->indices)) {
            std::cout << "picked " << indices_->indices.size () << std::endl;
        } else {
            std::cout << "No valid points selected!" << std::endl;
        }

        // Create the segmentation object
        pcl::SACSegmentationFromNormals<pcl::PointNormal, pcl::PointNormal> seg;
        pcl::PointIndices::Ptr cylinderinlierindices;
        pcl::ModelCoefficients::Ptr cylindercoeffs (new pcl::ModelCoefficients);
        // Optional
        seg.setOptimizeCoefficients (true);
        //seg.setAxis(...);
        // Mandatory
        seg.setModelType (pcl::SACMODEL_CYLINDER);
        seg.setMethodType (pcl::SAC_RANSAC);
        //seg.setMethodType (pcl::SAC_RRANSAC);
        seg.setMaxIterations(10000);
        seg.setDistanceThreshold (distthreshold_cylindermodel_);
        seg.setInputCloud (input_);
        seg.setInputNormals (input_);
        seg.segment (*cylinderinlierindices, *cylindercoeffs);

        std::cerr << "cylinder Model cylindercoeffs: " << cylindercoeffs->values[0] << " "
                  << cylindercoeffs->values[1] << " "
                  << cylindercoeffs->values[2] << " "
                  << cylindercoeffs->values[3] << " "
                  << cylindercoeffs->values[4] << " "
                  << cylindercoeffs->values[5] << " "
                  << cylindercoeffs->values[6] << " " << std::endl;

        viewer_->addCylinder(*cylindercoeffs);

        std::cout << "extract model" << std::endl;
        pcl::ExtractIndices<pcl::PointNormal> extract;
        pcl::PointCloud<pcl::PointNormal>::Ptr extractedpoints(new pcl::PointCloud<pcl::PointNormal>());
        extract.setInputCloud (input_);
        extract.setIndices (cylinderinlierindices);
        //extract.setNegative (true);
        extract.filter (*extractedpoints);
        pcl::io::savePCDFileBinaryCompressed ("extractedpoints.pcd", *extractedpoints);
    }/*}}}*/
    typename pcl::PointCloud<pcl::PointNormal>::Ptr input_;
    pcl::PointIndices::Ptr indices_;
    double distthreshold_cylindermodel_;
    double searchradius_;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;

};


int main (int argc, char** argv)
{
    // command line arguments parsing /*{{{*/
    namespace bpo = boost::program_options;
    namespace bfs = boost::filesystem;
    bpo::options_description opts_desc("Allowed options");
    bpo::positional_options_description p;

    opts_desc.add_options()
        ("input_file", bpo::value< std::string >(), "Input data.")
        ("output_file", bpo::value< std::string >(), "output data.")
        ("removeplane", bpo::value<bool>()->default_value(false), "findplane or not?")
        ("voxelsize", bpo::value< double >(), "voxelsize")
        ("distthreshold_findplane", bpo::value< double >(), "distance threshold for finding plane")
        ("cylindersurfacesearchradius", bpo::value< double >(), "")
        ("distthreshold_cylindermodel", bpo::value< double >(), "distance threshold for finding cylinder")
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
    std::string output_file = opts["output_file"].as<std::string> ();

    double voxelsize = opts["voxelsize"].as<double>();
    bool removeplane= opts["removeplane"].as<bool>();
    double distthreshold_findplane= opts["distthreshold_findplane"].as<double>();
    double cylindersurfacesearchradius= opts["cylindersurfacesearchradius"].as<double>();
    double distthreshold_cylindermodel= opts["distthreshold_cylindermodel"].as<double>();
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

    CableDetection cabledetection(cloud_normals, cylindersurfacesearchradius, distthreshold_cylindermodel);
    //pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colorcloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
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
    //}
    return (0);
}
