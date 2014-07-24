#ifndef __PCL_CABLE_DETECTION__
#define __PCL_CABLE_DETECTION__

#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/pcl_search.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <algorithm> //copy
#include <iterator> //back_inserter
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

template <typename PointNT>
class CableDetection {
public:
    class CableSlice {
public:
        //std::string name;
        pcl::ModelCoefficients::Ptr cylindercoeffs;
        pcl::PointIndices::Ptr cylinderindices;
        pcl::PointIndices::Ptr searchedindices;

        pcl::PointXYZ centerpt_;

        CableSlice() {
            cylindercoeffs.reset(new pcl::ModelCoefficients());
            cylinderindices.reset(new pcl::PointIndices());
            searchedindices.reset(new pcl::PointIndices());
        }
        virtual ~CableSlice(){
            cylindercoeffs.reset();
            cylinderindices.reset();
            searchedindices.reset();
        }
        //copy constructor
        //CableSlice(const CableSlice& sliceorig) {
        //cylindercoeffs = sliceorig.cylindercoeffs->;
        //}
        typedef boost::shared_ptr<CableSlice> Ptr;
        typedef boost::shared_ptr<CableSlice const> ConstPtr;
    };
    typedef boost::shared_ptr<CableSlice> CableSlicePtr;
    typedef boost::shared_ptr<CableSlice const> CableSliceConstPtr;

    //typedef std::list<CableSlice> Cable;
    typedef std::list<CableSlicePtr> Cable;

    typedef typename pcl::PointCloud<PointNT> PointCloudInput;
    typedef typename pcl::PointCloud<PointNT>::Ptr PointCloudInputPtr;
    CableDetection(PointCloudInputPtr input, double cableradius, double cableslicelen, double distthreshold_cylindermodel)
        : tryfindingpointscounts_(3)
    {
        input_ = input;
        // create pointcloud<pointxyz>
        points_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*input_, *points_);

        // Set up the full indices set
        // NOTE: do not remove points from input_
        // it will mess up indices_ and points_
        std::vector<int> full_indices (input_->points.size ());
        indices_.reset(new pcl::PointIndices());
        indices_->indices.resize(input_->points.size());
        for (int fii = 0; fii < static_cast<int> (indices_->indices.size ()); ++fii) {  // fii = full indices iterator
            indices_->indices[fii] = fii;
        }
        cableradius_ = cableradius;
        cableslicelen_ = cableslicelen;
        distthreshold_cylindermodel_ = distthreshold_cylindermodel;
        viewer_.reset(new pcl::visualization::PCLVisualizer("PCL viewer_"));
        viewer_->setBackgroundColor (0.0, 0.0, 0.0);
        viewer_->addPointCloud<PointNT> (input_, "inputcloud");
        //viewer_->registerAreaPickingCallback(boost::bind(&CableDetection::area_picking_callback, this,_1));
        viewer_->registerPointPickingCallback(boost::bind(&CableDetection::point_picking_callback, this, _1));
        viewer_->registerKeyboardCallback(boost::bind(&CableDetection::keyboard_callback, this, _1));
    }

    void RunViewer(){
        while (!viewer_->wasStopped ())
        {
            viewer_->spinOnce ();
        }
    }

    void keyboard_callback (const pcl::visualization::KeyboardEvent& event)
    {
        /*
           std::cout << event.getKeySym() << std::endl;
           if(event.getKeySym() == "i") {
            if (rendering_input_) {
                viewer_->removePointCloud("inputcloud");
                rendering_input_ = false;
            }
            else {
                viewer_->addPointCloud<pcl::PointNormal> (input_, "inputcloud");
                rendering_input_ = true;
            }
           }
         */
    }

    void point_picking_callback (const pcl::visualization::PointPickingEvent& event)
    {
        boost::mutex::scoped_lock lock(cables_mutex_);
        //pcl::PointXYZ pt;
        //event.getPoint (pt.x, pt.y, pt.z);
        size_t idx = event.getPointIndex ();
        std::cout << "picking point index: " << idx << std::endl;

        pcl::PointXYZ selectedpoint;
        selectedpoint.x = input_->points[idx].x;
        selectedpoint.y = input_->points[idx].y;
        selectedpoint.z = input_->points[idx].z;
        Cable cable = findCableFromPoint(selectedpoint);
        visualizeCable(cable);
    }

    Cable findCableFromPoint(pcl::PointXYZ point) {/*{{{*/
        pcl::PointIndices::Ptr k_indices;
        k_indices = findClosePointsIndices(point);
        pcl::PointXYZ pt;

        Cable cable;
        CableSlicePtr slice, oldslice, baseslice;
        slice.reset(new CableSlice());
        oldslice.reset(new CableSlice());
        bool cableslicefound = estimateCylinderAroundPointsIndices(k_indices, *slice);
        if (!cableslicefound) {
            PCL_INFO("[first search] no valid slice found\n");
            return cable;
        }
        oldslice = slice; baseslice = slice;
        std::cout << "first slice found!" << std::endl;
        cable.push_back(slice);

        // search forward
        for(size_t iteration=0;; iteration++) {
            int tryindex = tryfindingpointscounts_;
            pt.x = oldslice->cylindercoeffs->values[0];
            pt.y = oldslice->cylindercoeffs->values[1];
            pt.z = oldslice->cylindercoeffs->values[2];
            pt.z -= cableradius_; // make point closer to camera by cable radius

            bool pointsfound = false;
            while (true) {
                std::cout << "search forward! (try to extend: " << tryindex << ")" << std::endl;
                pt.x += oldslice->cylindercoeffs->values[3]*cableslicelen_;
                pt.y += oldslice->cylindercoeffs->values[4]*cableslicelen_;
                pt.z += oldslice->cylindercoeffs->values[5]*cableslicelen_;

                k_indices = findClosePointsIndices(pt);
                if (k_indices->indices.size() < 30) {
                    if (tryindex == 0) {
                        PCL_INFO("[forward search] very little (%d) close points found (itr:%d)\n", k_indices->indices.size(), iteration);
                        break;
                    } else {
                        PCL_INFO("[forward search] very little (%d) close points found (itr:%d)... but I don't give up.\n", k_indices->indices.size(), iteration);
                    }
                } else {
                    pointsfound = true;
                    break;
                }
                tryindex--;
            }
            if (!pointsfound) {
                break;
            }

            slice.reset(new CableSlice());
            slice->centerpt_= pt;
            Eigen::Vector3f initialaxis(oldslice->cylindercoeffs->values[3], oldslice->cylindercoeffs->values[4], oldslice->cylindercoeffs->values[5]);
            cableslicefound = estimateCylinderAroundPointsIndices(k_indices, *slice, pt, initialaxis, 1);
            if (!cableslicefound) {
                PCL_INFO("[forward search]: no valid slice found (itr:%d)\n", iteration);
                //NOTE: for debug
                std::copy(k_indices->indices.begin(), k_indices->indices.end(), std::back_inserter(slice->searchedindices->indices));
                cable.push_front(slice);
                break;
            }
            Eigen::Vector3f estimated_cylinder_axis(slice->cylindercoeffs->values[3],slice->cylindercoeffs->values[4],slice->cylindercoeffs->values[5]);
            if (initialaxis.dot(estimated_cylinder_axis) < 0 ) {
                slice->cylindercoeffs->values[3] = -slice->cylindercoeffs->values[3];
                slice->cylindercoeffs->values[4] = -slice->cylindercoeffs->values[4];
                slice->cylindercoeffs->values[5] = -slice->cylindercoeffs->values[5];
            }
            cable.push_front(slice);
            oldslice = slice;
        }

        oldslice = baseslice;
        // search backward
        for(size_t iteration=0;; iteration++) {
            int tryindex = tryfindingpointscounts_;
            pt.x = oldslice->cylindercoeffs->values[0];
            pt.y = oldslice->cylindercoeffs->values[1];
            pt.z = oldslice->cylindercoeffs->values[2];
            pt.z -= cableradius_; // make point closer to camera by cable radius

            bool pointsfound = false;
            while (true) {
                std::cout << "search backward! (try to extend: " << tryindex << ")" << std::endl;
                pt.x -= oldslice->cylindercoeffs->values[3]*cableslicelen_;
                pt.y -= oldslice->cylindercoeffs->values[4]*cableslicelen_;
                pt.z -= oldslice->cylindercoeffs->values[5]*cableslicelen_;

                k_indices = findClosePointsIndices(pt);
                if (k_indices->indices.size() < 30) {
                    if (tryindex == 0) {
                        PCL_INFO("[forward search] very little (%d) close points found (itr:%d)\n", k_indices->indices.size(), iteration);
                        break;
                    } else {
                        PCL_INFO("[forward search] very little (%d) close points found (itr:%d)... but I don't give up.\n", k_indices->indices.size(), iteration);
                    }
                } else {
                    pointsfound = true;
                    break;
                }
                tryindex--;
            }
            if (!pointsfound) {
                break;
            }

            slice.reset(new CableSlice());
            slice->centerpt_ = pt;
            Eigen::Vector3f initialaxis(oldslice->cylindercoeffs->values[3], oldslice->cylindercoeffs->values[4], oldslice->cylindercoeffs->values[5]);
            cableslicefound = estimateCylinderAroundPointsIndices(k_indices, *slice, pt, initialaxis, 1);
            if (!cableslicefound) {
                PCL_INFO("[backward search]: no valid slice found (itr:%d)\n", iteration);
                //NOTE: for debug
                std::copy(k_indices->indices.begin(), k_indices->indices.end(), std::back_inserter(slice->searchedindices->indices));
                cable.push_back(slice);
                break;
            }
            Eigen::Vector3f estimated_cylinder_axis(slice->cylindercoeffs->values[3],slice->cylindercoeffs->values[4],slice->cylindercoeffs->values[5]);
            if (initialaxis.dot(estimated_cylinder_axis) < 0 ) {
                slice->cylindercoeffs->values[3] = -slice->cylindercoeffs->values[3];
                slice->cylindercoeffs->values[4] = -slice->cylindercoeffs->values[4];
                slice->cylindercoeffs->values[5] = -slice->cylindercoeffs->values[5];
            }
            cable.push_back(slice);
            oldslice = slice;
        }

        return cable;
    }/*}}}*/

    void visualizeCable(Cable& cable) {/*{{{*/
        viewer_->removeAllShapes();
        size_t sliceindex = 0;
        for (typename std::list<CableSlicePtr>::iterator itr = cable.begin(); itr != cable.end(); ++itr, ++sliceindex) {
            if (sliceindex == cable.size()-1) {
                break;
            }
            std::stringstream ss;
            ss << "cylinder_" << sliceindex;
            std::string cylindername = ss.str();
            //viewer_->addCylinder(*cylindercoeffs);
            pcl::PointXYZ pt0, pt1;
            pt0.x = (*itr)->cylindercoeffs->values[0];
            pt0.y = (*itr)->cylindercoeffs->values[1];
            pt0.z = (*itr)->cylindercoeffs->values[2];
            pt1.x = (*boost::next(itr))->cylindercoeffs->values[0];
            pt1.y = (*boost::next(itr))->cylindercoeffs->values[1];
            pt1.z = (*boost::next(itr))->cylindercoeffs->values[2];

            viewer_->removeShape(cylindername);
            //viewer_->addLine(pt0, pt1,(sliceindex%3==0?1:0),((sliceindex+1)%3==0?1:0),((sliceindex+2)%3==0?1:0), cylindername);
            int r = (sliceindex%3==0 ? 1 : 0);
            int g = ((sliceindex+1)%3==0 ? 1 : 0);
            int b = ((sliceindex+2)%3==0 ? 1 : 0);
            viewer_->addArrow(pt0, pt1, r, g, b, false, cylindername);

            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr extractedpoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            pcl::copyPointCloud (*input_, *((*itr)->searchedindices), *extractedpoints);

            uint32_t rgb = (static_cast<uint32_t>(r*255) << 16 | static_cast<uint32_t>(g*255) << 8 | static_cast<uint32_t>(b*255));
            for (size_t i = 0; i < extractedpoints->points.size(); i++) {
                extractedpoints->points[i].rgb = *reinterpret_cast<float*>(&rgb);
            }


            std::stringstream slicepointsid;
            slicepointsid << "slicepoints_" << sliceindex;
            viewer_->removePointCloud(slicepointsid.str());
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgbfield(extractedpoints);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBNormal> rgbfield(extractedpoints, r*255, g*255, b*255);
            viewer_->addPointCloud<pcl::PointXYZRGBNormal> (extractedpoints, rgbfield, slicepointsid.str());
            viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, slicepointsid.str());
            //viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0,0.0,1.0, "sample cloud_2");
            //viewer_->addPointCloud(extractedpoints, slicepointsid.str());
            ////const std::string slicepointsidstr = slicepointsid.str();
            ////viewer_->addPointCloudNormals (extractedpoints, 100, 0.002f, slicepointsidstr,0);
            //
            std::stringstream slicecubeid;
            slicecubeid << "slicesearchcube_" << sliceindex;
            viewer_->addSphere((*itr)->centerpt_, cableradius_*2, (r*255), (g*255), (b*255), slicecubeid.str());
            std::cout << "slice " << sliceindex << " points size: " << extractedpoints->points.size() << std::endl;

            std::stringstream textss;
            textss << "slice_" << sliceindex;
            viewer_->addText3D (textss.str(), pt0, 0.001);
        }

        //if (extractedpoints->points.size() > 0) {
        //    pcl::io::savePCDFileBinaryCompressed ("extractedpoints.pcd", *extractedpoints);
        //}
    }/*}}}*/

    pcl::PointIndices::Ptr findClosePointsIndices(pcl::PointXYZ pt, double radius = 0) { /*{{{*/
        if (radius == 0) {
            radius = cableradius_*2;
        }
        pcl::PointIndices::Ptr k_indices(new pcl::PointIndices());
        std::vector<float> k_sqr_distances;
        pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZ>());
        // NOTE: when you change input_, you also need to change points_!
        tree->setInputCloud(points_);
        tree->radiusSearch (pt, radius, k_indices->indices, k_sqr_distances);
        return k_indices;
    } /*}}}*/

    bool estimateCylinderAroundPointsIndices (pcl::PointIndices::Ptr pointsindices, CableSlice& slice, pcl::PointXYZ centerpt = pcl::PointXYZ(), const Eigen::Vector3f& initialaxis = Eigen::Vector3f(), double eps_angle=0.0) /*{{{*/
    {
        // Create the segmentation object
        pcl::SACSegmentationFromNormals<PointNT, PointNT> seg;
        pcl::PointIndices::Ptr cylinderinlierindices(new pcl::PointIndices());
        // Optional
        seg.setOptimizeCoefficients (true);
        seg.setAxis (initialaxis);
        seg.setEpsAngle(eps_angle);
        // Mandatory
        seg.setModelType (pcl::SACMODEL_CYLINDER);
        seg.setMethodType (pcl::SAC_RANSAC);
        //seg.setMethodType (pcl::SAC_RRANSAC);
        seg.setMaxIterations(10000);
        seg.setDistanceThreshold (distthreshold_cylindermodel_);
        seg.setInputCloud (input_);
        seg.setInputNormals (input_);
        seg.setIndices(pointsindices);
        seg.segment (*slice.cylinderindices, *slice.cylindercoeffs);

        if (!(centerpt.x == 0 && centerpt.y == 0 && centerpt.z == 0)) {
            // fix the center of the slice
            Eigen::Vector3f pos(slice.cylindercoeffs->values[0],slice.cylindercoeffs->values[1],slice.cylindercoeffs->values[2]);
            Eigen::Vector3f w(centerpt.x - slice.cylindercoeffs->values[0], centerpt.y - slice.cylindercoeffs->values[1], centerpt.z - slice.cylindercoeffs->values[2]);
            Eigen::Vector3f dir(slice.cylindercoeffs->values[3],slice.cylindercoeffs->values[4],slice.cylindercoeffs->values[5]); //dir should be normalized
            //orthogonal projection to the line
            Eigen::Vector3f newpos = pos + w.dot(dir) * dir;

            slice.cylindercoeffs->values[0] = newpos[0];
            slice.cylindercoeffs->values[1] = newpos[1];
            slice.cylindercoeffs->values[2] = newpos[2];
        }

        std::copy(pointsindices->indices.begin(), pointsindices->indices.end(), std::back_inserter(slice.searchedindices->indices) );
        std::cerr << "cylinder Model cylindercoeffs: "
                  << slice.cylindercoeffs->values[0] << " "
                  << slice.cylindercoeffs->values[1] << " "
                  << slice.cylindercoeffs->values[2] << " "
                  << slice.cylindercoeffs->values[3] << " "
                  << slice.cylindercoeffs->values[4] << " "
                  << slice.cylindercoeffs->values[5] << " "
                  << slice.cylindercoeffs->values[6] << " " << std::endl;
        return validateCableSlice(slice.cylindercoeffs);
    } /*}}}*/

    bool validateCableSlice (pcl::ModelCoefficients::Ptr cylindercoeffs) /*{{{*/
    {
        PCL_INFO("[validateCableSlice] radius: %f, given: %f\n", cylindercoeffs->values[6], cableradius_);
        if(cableradius_* 0.7 < cylindercoeffs->values[6] && cylindercoeffs->values[6] < cableradius_* 1.3 ) {
            return true;
        }
        return false;
    } /*}}}*/

    void area_picking_callback (const pcl::visualization::AreaPickingEvent &event) /*{{{*/
    {
        if (event.getPointsIndices (indices_->indices)) {
            std::cout << "picked " << indices_->indices.size () << std::endl;
        } else {
            std::cout << "No valid points selected!" << std::endl;
        }

        // Create the segmentation object
        pcl::SACSegmentationFromNormals<PointNT, PointNT> seg;
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
        pcl::ExtractIndices<PointNT> extract;
        PointCloudInputPtr extractedpoints(new PointCloudInput());
        extract.setInputCloud (input_);
        extract.setIndices (cylinderinlierindices);
        //extract.setNegative (true);
        extract.filter (*extractedpoints);
        pcl::io::savePCDFileBinaryCompressed ("extractedpoints.pcd", *extractedpoints);
    } /*}}}*/

    PointCloudInputPtr input_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_;
    pcl::PointIndices::Ptr indices_;
    double distthreshold_cylindermodel_;
    double cableradius_;
    double cableslicelen_;
    std::vector<Cable> cables_;
    const int tryfindingpointscounts_;

    boost::mutex cables_mutex_;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;

};


} // namespace pcl_cable_detection
#endif /* end of include guard */

