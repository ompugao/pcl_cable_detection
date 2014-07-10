#ifndef __PCL_CABLE_DETECTION__
#define __PCL_CABLE_DETECTION__

#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <cstdlib> //random

namespace pcl_cable_detection {

/** \brief find plane
 * \param[in] cloud the PointCloud dataset
 * \param[in] voxelleafsize the size of voxelleaf
 * \param[in] distthreshold the distance threshold to determine whether each point belongs to plane or not
 * \param[out] inlierindices inlier indices of the points which is contained inside the plane
 * \param[out] modelcoeffs coefficients of the plane
 */
template<typename PointT>
bool findPlane(const typename pcl::PointCloud<PointT>::Ptr cloud,
               double voxelleafsize,
               double distthreshold,
               pcl::PointIndices& inlierindices,
               pcl::ModelCoefficients& modelcoeffs
               )
{
    // Create voxel grid
    pcl::VoxelGrid<PointT> vg;
    typename pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
    vg.setInputCloud (cloud);
    vg.setLeafSize (voxelleafsize, voxelleafsize, voxelleafsize);
    vg.filter (*cloud_filtered);


    // Create the segmentation object
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices voxelinlierindices;
    // Optional
    seg.setOptimizeCoefficients (true);
    //seg.setAxis(...);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    //seg.setMethodType (pcl::SAC_RRANSAC);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold (distthreshold);
    seg.setInputCloud (cloud_filtered);
    seg.segment (voxelinlierindices, modelcoeffs);

    if (voxelinlierindices.indices.size () == 0)
    {
        return false;
    }

    // Find the distance from point to plane.
    // http://mathworld.wolfram.com/Point-PlaneDistance.html
    double denominator = sqrt(pow(modelcoeffs.values[0], 2) + pow(modelcoeffs.values[1], 2) + pow(modelcoeffs.values[2], 2));
    for (size_t i = 0; i < cloud->size(); i++) {
        double dist = cloud->points[i].x * modelcoeffs.values[0] + cloud->points[i].y * modelcoeffs.values[1] +  cloud->points[i].z * modelcoeffs.values[2] + modelcoeffs.values[3];
        dist /=  denominator;
        dist = (dist >= 0) ? dist : -dist;
        if (dist < distthreshold) {
            inlierindices.indices.push_back(i);
        }
    }

    return true;
}

/** \brief compute curvature histogram for each point using a given radius
 */
template<typename PointT, int N>
void computeCurvatureHistogram(const typename pcl::PointCloud<PointT>::Ptr cloud,
                               int pointindex,
                               double radius, //pcl::PointIndices::Ptr indices,
                               float min,
                               float max,
                               pcl::Histogram<N>& histogram
                               )
{
    typedef typename pcl::search::KdTree<PointT> KdTree;
    typedef typename pcl::search::KdTree<PointT>::Ptr KdTreePtr;
    std::vector<int> k_indices;
    std::vector<float> k_sqr_distances;
    KdTreePtr tree (new KdTree());
    tree->setInputCloud(cloud);
    tree->radiusSearch (cloud->points[index], radius, k_indices, k_sqr_distances);

    float binwidth = (max - min) * 1.0 / N;
    for (std::vector<int>::const_iterator itr = k_indices.begin(); itr != k_indices.end(); ++itr) {
        histogram[(cloud->points[*itr].curvature)/binwidth] += 1;
    }
}

} // namespace pcl_cable_detection
#endif /* end of include guard */

