#ifndef __PCL_CABLE_DETECTION__
#define __PCL_CABLE_DETECTION__

#include <boost/random.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/io.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/pcl_search.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <algorithm> //copy
#include <iterator> //back_inserter
#include <cstdlib> //random
#include <cmath>
#include <vtkTransform.h>

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
    class CableTerminal;
    typedef boost::shared_ptr<CableTerminal> CableTerminalPtr;
    typedef boost::shared_ptr<CableTerminal const> CableTerminalConstPtr;

    class CableSlice {
public:
        //std::string name;
        pcl::ModelCoefficients::Ptr cylindercoeffs;
        pcl::PointIndices::Ptr cylinderindices;
        pcl::PointIndices::Ptr searchedindices;
        CableTerminalPtr cableterminal;

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

    class CableTerminal {
public:
        Eigen::Matrix4f transform;

        CableTerminal() {
        }
        virtual ~CableTerminal(){
        }

        typedef boost::shared_ptr<CableSlice> Ptr;
        typedef boost::shared_ptr<CableSlice const> ConstPtr;
public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };



    typedef typename pcl::PointCloud<PointNT> PointCloudInput;
    typedef typename pcl::PointCloud<PointNT>::Ptr PointCloudInputPtr;
    typedef pcl::FPFHSignature33 FeatureT;
    typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
    typedef pcl::PointCloud<FeatureT> FeatureCloudT;

    CableDetection(PointCloudInputPtr terminalcloud, Eigen::Vector3f terminalaxis)
        : tryfindingpointscounts_(3)
    {
        cableradius_ = 0;
        distthreshold_cylindermodel_ = 0;
        scenesampling_radius_ = 0;

        // compute the bounding cylinder of the terminal
        terminalcloud_ = terminalcloud;
        terminalcylindercoeffs_.reset(new pcl::ModelCoefficients());
        computeBoundingCylinder(terminalcloud_, terminalaxis, terminalcylindercoeffs_);
    }

    void setInputCloud(PointCloudInputPtr input)
    {
        input_ = input;
        if (!!viewer_) {
            boost::mutex::scoped_lock lock(viewer_mutex_);
            viewer_->removePointCloud("inputcloud");
            viewer_->addPointCloud<PointNT> (input_, "inputcloud");
        }
        // create pointcloud<pointxyz>
        points_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*input_, *points_);
        // setup kdtree
        kdtree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>());
        kdtree_->setInputCloud(points_);

        // Set up the full indices set
        // NOTE: do not remove points from input_
        // it will mess up indices_ and points_
        std::vector<int> full_indices (input_->points.size ());
        indices_.reset(new pcl::PointIndices());
        indices_->indices.resize(input_->points.size());
        for (int fii = 0; fii < static_cast<int> (indices_->indices.size ()); ++fii) {  // fii = full indices iterator
            indices_->indices[fii] = fii;
        }

    }

    void setCableRadius(double cableradius) {
        cableradius_ = cableradius;
        cableslicelen_ = 4 * cableradius_;
    }
    void setThresholdCylinderModel(double distthreshold_cylindermodel) {
        distthreshold_cylindermodel_ = distthreshold_cylindermodel;
    }
    void setSceneSamplingRadius(double scenesampling_radius) {
        scenesampling_radius_ = scenesampling_radius;
    }

    void LockViewer() {
        viewer_mutex_.lock();
    }
    void UnLockViewer() {
        viewer_mutex_.unlock();
    }

    void SetUpViewer() { /*{{{*/
        {
            boost::mutex::scoped_lock lock(viewer_mutex_);
            viewer_.reset(new pcl::visualization::PCLVisualizer("PCL cable detection viewer"));
            viewer_->setBackgroundColor (0.0, 0.0, 0.0);
            viewer_->addCoordinateSystem(0.5);
            //viewer_->registerAreaPickingCallback(boost::bind(&CableDetection::area_picking_callback, this,_1));
            viewer_->registerPointPickingCallback(boost::bind(&CableDetection::point_picking_callback, this, _1));
            viewer_->registerKeyboardCallback(boost::bind(&CableDetection::keyboard_callback, this, _1));
        }
        /*
        std::vector<Cable> cables;
        {
            pcl::ScopeTime t("findCables");
            findCables(cables);
        }
        int i = 0;
        for (typename std::vector<Cable>::iterator itr = cables.begin(); itr != cables.end(); itr++ , i++) {
            std::stringstream ss;
            ss << "cable" << i << "_";
            visualizeCable(*itr, ss.str());
            viewer_->spinOnce ();
        }
        std::cout << "found " << cables.size() << " cables"<< std::endl;
        */
    } /*}}}*/
    void RunViewer() {/*{{{*/
        if(!viewer_) {
            SetUpViewer();
        }
        while (!viewer_->wasStopped ())
        {
            {
                boost::mutex::scoped_lock lock(viewer_mutex_);
                viewer_->spinOnce ();
            }
            boost::this_thread::sleep(boost::posix_time::milliseconds(100));
        }
    }/*}}}*/
    void RunViewerBackGround(){ /*{{{*/
        if(!viewer_) {
            SetUpViewer();
        }
        viewerthread_.reset(new boost::thread(boost::bind(&CableDetection::RunViewer, this)));
    }/*}}}*/

    /*
     * param[in] cloud: model cloud
     * param[in] axis: bounding cylinder axis of the model
     * param[out] cylindercoeffs: the extended cylinder model coefficients
     */
    void computeBoundingCylinder(PointCloudInputPtr cloud, Eigen::Vector3f axis, pcl::ModelCoefficients::Ptr cylindercoeffs) { /*{{{*/
        cylindercoeffs->values.resize(9);
        cylindercoeffs->values[0] = cylindercoeffs->values[1] = cylindercoeffs->values[2] = 0.0;
        axis.normalize();
        cylindercoeffs->values[3] = axis[0];
        cylindercoeffs->values[4] = axis[1];
        cylindercoeffs->values[5] = axis[2];

        double square_r = 0;
        double upperheight = 0;
        double lowerheight = std::numeric_limits<float>::max();
        for (size_t i = 0; i < terminalcloud_->points.size(); i++) {
            PointNT& pt = terminalcloud_->points[i];
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            double cos = p.dot(axis);
            Eigen::Vector3f projected_p = cos * axis;
            double tmp_height = projected_p.norm();
            double tmp_square_r = (p - projected_p).squaredNorm();
            if (square_r < tmp_square_r) { square_r = tmp_square_r; }
            if (cos > 0) {
                if (upperheight < tmp_height) { upperheight = tmp_height; }
            } else {
                if (lowerheight > -tmp_height) { lowerheight = -tmp_height; }
            }
        }
        cylindercoeffs->values[6] = std::sqrt(square_r);
        cylindercoeffs->values[7] = upperheight;
        cylindercoeffs->values[8] = lowerheight;
    } /*}}}*/

    void point_picking_callback (const pcl::visualization::PointPickingEvent& event) /*{{{*/
    {
        //pcl::PointXYZ pt;
        //event.getPoint (pt.x, pt.y, pt.z);
        size_t idx = event.getPointIndex ();
        std::cout << "picking point index: " << idx << std::endl;

        pcl::PointXYZ selectedpoint;
        selectedpoint.x = input_->points[idx].x;
        selectedpoint.y = input_->points[idx].y;
        selectedpoint.z = input_->points[idx].z;
        Cable cable = findCableFromPoint(selectedpoint);
        // do not lock viewer mutex!!! the function who calls point_picking_callback itself is locking mutex
        viewer_->removeAllShapes();
        visualizeCable(cable);
        //findCableTerminal(cable, 0.027);
        //findCableTerminal(cable, 0.018);
    } /*}}}*/

    // lock viewer occasionally
    void findCables(std::vector<Cable>& cables)
    {
        pcl::PointCloud<int> sampled_indices;
        pcl::UniformSampling<PointNT> uniform_sampling;
        uniform_sampling.setInputCloud (input_);
        uniform_sampling.setRadiusSearch (scenesampling_radius_);
        uniform_sampling.compute (sampled_indices);
        PointCloudInputPtr sampledpoints(new PointCloudInput);
        pcl::copyPointCloud (*input_, sampled_indices.points, *sampledpoints);
        if (!!viewer_) {
            boost::mutex::scoped_lock lock(viewer_mutex_);
            viewer_->removePointCloud ("sampledpoints");
            pcl::visualization::PointCloudColorHandlerCustom<PointNT> rgbfield(sampledpoints, 255, 128, 128);
            viewer_->addPointCloud<PointNT> (sampledpoints, rgbfield,  "sampledpoints");
            viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "sampledpoints");
        }

        for (size_t isample = 0; isample < sampled_indices.points.size(); isample++) {
            bool alreadyscanned = false;

            // check if the sampled point is already found or not.
            {
            pcl::ScopeTime t("<----> sampled point is already found? <---->");
            pcl::PointIndices::Ptr indicesaround = findClosePointsIndices(points_->points[isample]);
            for (std::vector<int>::iterator i = indicesaround->indices.begin(); i != indicesaround->indices.end(); i++) {
                for (typename std::vector<Cable>::iterator itr = cables.begin(); itr != cables.end(); itr++) {
                    for (typename std::list<CableSlicePtr>::iterator sliceitr = (*itr).begin(); sliceitr!= (*itr).end(); ++sliceitr) {
                        std::vector<int>::iterator founditr 
                            = std::find( (*sliceitr)->searchedindices->indices.begin(), (*sliceitr)->searchedindices->indices.end() , (*i) );
                        if ( founditr !=(*sliceitr)->searchedindices->indices.end() ) {
                            alreadyscanned = true;
                            break;
                        }
                    }
                }
            }
            }
            if (not alreadyscanned) {
                Cable cable = findCableFromPoint(points_->points[isample]);
                if(cable.size() > 0) {
                    cables.push_back(cable);
                }
            }
        }
    }

    // note: viewer lock free
    Cable findCableFromPoint(pcl::PointXYZ point) { /*{{{*/
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
                //std::copy(k_indices->indices.begin(), k_indices->indices.end(), std::back_inserter(slice->searchedindices->indices));
                //cable.push_front(slice);
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
                //std::copy(k_indices->indices.begin(), k_indices->indices.end(), std::back_inserter(slice->searchedindices->indices));
                //cable.push_back(slice);
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
    } /*}}}*/

    // note: lock viewer_mutex_ beforehand
    void visualizeCable(Cable& cable, std::string namesuffix = "") { /*{{{*/
        size_t sliceindex = 0;
        for (typename std::list<CableSlicePtr>::iterator itr = cable.begin(); itr != cable.end(); ++itr, ++sliceindex) {
            if (sliceindex == cable.size()-1) {
                break;
            }
            std::stringstream ss;
            ss << namesuffix << "cylinder_" << sliceindex;
            std::string cylindername = ss.str();
            //viewer_->addCylinder(*cylindercoeffs);
            pcl::PointXYZ pt0, pt1;
            pt0.x = (*itr)->cylindercoeffs->values[0];
            pt0.y = (*itr)->cylindercoeffs->values[1];
            pt0.z = (*itr)->cylindercoeffs->values[2];
            pt1.x = (*boost::next(itr))->cylindercoeffs->values[0];
            pt1.y = (*boost::next(itr))->cylindercoeffs->values[1];
            pt1.z = (*boost::next(itr))->cylindercoeffs->values[2];

            //viewer_->removeShape(cylindername);
            //viewer_->addLine(pt0, pt1,(sliceindex%3==0?1:0),((sliceindex+1)%3==0?1:0),((sliceindex+2)%3==0?1:0), cylindername);
            int r = (sliceindex%3==0 ? 1 : 0);
            int g = ((sliceindex+1)%3==0 ? 1 : 0);
            int b = ((sliceindex+2)%3==0 ? 1 : 0);
            //viewer_->addArrow(pt0, pt1, r, g, b, false, cylindername);

            pt0.x = (*itr)->cylindercoeffs->values[0];
            pt0.y = (*itr)->cylindercoeffs->values[1];
            pt0.z = (*itr)->cylindercoeffs->values[2];
            pt1.x = (*itr)->cylindercoeffs->values[0] + (*itr)->cylindercoeffs->values[3]*cableslicelen_;
            pt1.y = (*itr)->cylindercoeffs->values[1] + (*itr)->cylindercoeffs->values[4]*cableslicelen_;
            pt1.z = (*itr)->cylindercoeffs->values[2] + (*itr)->cylindercoeffs->values[5]*cableslicelen_;

            viewer_->addArrow(pt0, pt1, r, g, b, false, cylindername);

            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr extractedpoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            pcl::copyPointCloud (*input_, *((*itr)->searchedindices), *extractedpoints);

            uint32_t rgb = (static_cast<uint32_t>(r*255) << 16 | static_cast<uint32_t>(g*255) << 8 | static_cast<uint32_t>(b*255));
            for (size_t i = 0; i < extractedpoints->points.size(); i++) {
                extractedpoints->points[i].rgb = *reinterpret_cast<float*>(&rgb);
            }


            std::stringstream slicepointsid;
            slicepointsid <<namesuffix<< "slicepoints_" << sliceindex;
            viewer_->removePointCloud(slicepointsid.str());
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBNormal> rgbfield(extractedpoints, r*255, g*255, b*255);
            viewer_->addPointCloud<pcl::PointXYZRGBNormal> (extractedpoints, rgbfield, slicepointsid.str());
            viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, slicepointsid.str());

            //viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0,0.0,1.0, "sample cloud_2");
            //viewer_->addPointCloud(extractedpoints, slicepointsid.str());
            ////const std::string slicepointsidstr = slicepointsid.str();
            ////viewer_->addPointCloudNormals (extractedpoints, 100, 0.002f, slicepointsidstr,0);

            //std::stringstream slicesphereid;
            //slicesphereid << "slicesearchsphere_" << sliceindex;
            //viewer_->addSphere((*itr)->centerpt_, cableradius_*2, (r*255), (g*255), (b*255), slicesphereid.str());
            std::cout << "slice " << sliceindex
                << " Inliers: " << (*itr)->cylinderindices->indices.size() << "/"
                << (*itr)->searchedindices->indices.size()
                << std::endl;

            std::stringstream textss;
            textss << namesuffix << "slice_" << sliceindex;
            viewer_->addText3D (textss.str(), pt0, 0.001);
        }
    } /*}}}*/

    void findCableTerminal(Cable& cable, double offset) {
        if (cable.size() < 3) {
            pcl::console::print_highlight("no enough cable slices to find terminals. break.");
            return;
        }
        //CableSlicePtr firstslice = cable.front();
        //CableSlicePtr lastslice  = cable.back();
        //
        pcl::ModelCoefficients::Ptr firstterminalcoeffs(new pcl::ModelCoefficients(*terminalcylindercoeffs_));
        pcl::PointXYZ pt;
        Eigen::Vector3f dir; 
        dir(0) = (*cable.begin())->cylindercoeffs->values[3] + (*boost::next(cable.begin(),1))->cylindercoeffs->values[3] + (*boost::next(cable.begin(),2))->cylindercoeffs->values[3];
        dir(1) = (*cable.begin())->cylindercoeffs->values[4] + (*boost::next(cable.begin(),1))->cylindercoeffs->values[4] + (*boost::next(cable.begin(),2))->cylindercoeffs->values[4];
        dir(2) = (*cable.begin())->cylindercoeffs->values[5] + (*boost::next(cable.begin(),1))->cylindercoeffs->values[5] + (*boost::next(cable.begin(),2))->cylindercoeffs->values[5];
        dir.normalize();
        firstterminalcoeffs->values[0] = (*cable.begin())->cylindercoeffs->values[0] + dir(0) * (offset);
        firstterminalcoeffs->values[1] = (*cable.begin())->cylindercoeffs->values[1] + dir(1) * (offset);
        firstterminalcoeffs->values[2] = (*cable.begin())->cylindercoeffs->values[2] + dir(2) * (offset);
        firstterminalcoeffs->values[3] = dir[0];
        firstterminalcoeffs->values[4] = dir[1];
        firstterminalcoeffs->values[5] = dir[2];
        firstterminalcoeffs->values[6] += 0.005;
        firstterminalcoeffs->values[7] += 0.003;
        firstterminalcoeffs->values[8] -= 0.005;

        pcl::PointIndices::Ptr firstterminalscenepointsindices(new pcl::PointIndices());
        size_t points = findScenePointIndicesInsideCylinder(firstterminalcoeffs, firstterminalscenepointsindices);
        if (points < 30) {
            PCL_INFO("too few points at the first slice to detect termianal\n");
        } else {
            pcl::ExtractIndices<PointNT> extract;
            PointCloudInputPtr firstterminalscenepoints(new PointCloudInput());
            extract.setInputCloud (input_);
            extract.setIndices (firstterminalscenepointsindices);
            //extract.setNegative (true);
            extract.filter (*firstterminalscenepoints);

            // visualize points/*{{{*/
            //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgbfield(extractedpoints);
            pcl::visualization::PointCloudColorHandlerCustom<PointNT> rgbfield(firstterminalscenepoints, 255, 0, 255);

            viewer_->removePointCloud ("extracted_points");
            viewer_->addPointCloud<PointNT> (firstterminalscenepoints, rgbfield,  "extracted_points");
            viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "extracted_points");
            pcl::ModelCoefficients::Ptr cylmodel(new pcl::ModelCoefficients());
            cylmodel->values.resize(7);
            for (int i = 0; i < 7; i++) {
                cylmodel->values[i] = firstterminalcoeffs->values[i];
            }
            std::cout << firstterminalcoeffs->values[0] << " "
                << firstterminalcoeffs->values[1] << " "
                << firstterminalcoeffs->values[2] << " " << std::endl;
            std::cout << firstterminalcoeffs->values[3] << " "
                << firstterminalcoeffs->values[4] << " "
                << firstterminalcoeffs->values[5] << " " << std::endl;
            std::cout << firstterminalcoeffs->values[6] << " "
                << firstterminalcoeffs->values[7] << " "
                << firstterminalcoeffs->values[8] << " " << std::endl;
            //viewer_->addCylinder(*cylmodel);
            // end of visualize points/*}}}*/

            pcl::console::print_highlight ("Downsampling...\n");
            pcl::VoxelGrid<PointNT> grid;
            PointCloudInputPtr voxelterminalcloud(new PointCloudInput());
            PointCloudInputPtr voxelterminalcloudclosetoscene(new PointCloudInput());
            double leafsize = 0.001;
            grid.setLeafSize (leafsize, leafsize, leafsize);
            grid.setInputCloud (firstterminalscenepoints);
            grid.filter (*firstterminalscenepoints);
            grid.setInputCloud (terminalcloud_);
            grid.filter (*voxelterminalcloud);

            // Estimate normals for scene
            pcl::console::print_highlight ("Estimating scene normals...\n");
            pcl::NormalEstimationOMP<PointNT,PointNT> nest;
            nest.setRadiusSearch (2*leafsize);
            nest.setInputCloud (firstterminalscenepoints);
            nest.compute (*firstterminalscenepoints);
            nest.setInputCloud (voxelterminalcloud);
            nest.compute (*voxelterminalcloud);

            // compute cross product
            Eigen::Vector3f v(terminalcylindercoeffs_->values[3],terminalcylindercoeffs_->values[4],terminalcylindercoeffs_->values[5]);
            Eigen::Vector3f rotaxis = v.cross(dir);
            double rottheta = asin(rotaxis.norm());
            rotaxis.normalize();
            //Eigen::Transform<float, 3, Eigen::Affine> t; //same as the following one
            Eigen::Affine3f t;
            t = Eigen::Translation<float, 3>(firstterminalcoeffs->values[0], firstterminalcoeffs->values[1], firstterminalcoeffs->values[2]) * Eigen::AngleAxisf(rottheta, rotaxis);
            viewer_->addCoordinateSystem(0.1, t);

            FeatureCloudT::Ptr terminal_features (new FeatureCloudT); //should be a class member
            FeatureCloudT::Ptr scene_features (new FeatureCloudT); //should be a class member
            FeatureEstimationT fest;
            
            fest.setRadiusSearch (0.002);
            fest.setInputCloud (firstterminalscenepoints);
            fest.setInputNormals (firstterminalscenepoints);
            fest.compute (*scene_features);
            fest.setInputCloud (voxelterminalcloud);
            fest.setInputNormals (voxelterminalcloud);
            fest.compute (*terminal_features);
            

            //////////////////////////////////
/*{{{*/
            /*
            pcl::IterativeClosestPointWithNormals<PointNT, PointNT> icp;
            size_t rotnum = 4;
            for (size_t irot = 0; irot < rotnum; irot++) {
                pcl::console::print_highlight ("Starting terminal alignment... (%d)\n", irot);
                Eigen::Affine3f rotaffine;
                rotaffine = Eigen::AngleAxisf(2.0*M_PI/rotnum,v);
                pcl::transformPointCloudWithNormals (*voxelterminalcloud, *voxelterminalcloud, rotaffine);
                icp.setInputCloud(firstterminalscenepoints);
                icp.setInputTarget(voxelterminalcloud);
                // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
                icp.setMaxCorrespondenceDistance (0.01);
                // Set the maximum number of iterations (criterion 1)
                icp.setMaximumIterations (1000);
                // Set the transformation epsilon (criterion 2)
                icp.setTransformationEpsilon (1e-8);
                // Set the euclidean distance difference epsilon (criterion 3)
                icp.setEuclideanFitnessEpsilon (1);

                PointCloudInput finalpoints;
                icp.align(finalpoints);
                std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;

                if (icp.hasConverged ())
                {
                    // Print results
                    printf ("\n");
                    Eigen::Matrix4f transformation = icp.getFinalTransformation ();
                    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
                    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
                    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
                    pcl::console::print_info ("\n");
                    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
                    pcl::console::print_info ("\n");
                    //pcl::console::print_info ("Inliers: %i/%i\n", icp.getInliers ().size (), voxelterminalcloud->size ());

                    // Show alignment
                    PointCloudInputPtr terminalcloudforvis(new PointCloudInput());
                    //pcl::visualization::PointCloudColorHandlerRGBField<PointNT> rgbfield(terminalcloud_);
                    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBNormal> rgbfield(extractedpoints, 0, 255, 0);
                    //pcl::copyPointCloud(terminalcloud_, terminalcloudforvis);

                    //pcl::transformPointCloudWithNormals (*terminalcloud_, *terminalcloudforvis, transformation);
                    //pcl::visualization::PointCloudColorHandlerCustom<PointNT> rgbfield(object_aligned, 0, 255, 0);
                    //viewer_->addPointCloud<PointNT> (terminalcloudforvis, rgbfield, "firstterminal");
                    ////viewer_->addPointCloud<PointNT> (object_aligned, rgbfield, "firstterminal");
                    //viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "firstterminal");

                    vtkSmartPointer<vtkMatrix4x4> vtkmatrix = vtkSmartPointer<vtkMatrix4x4>::New();
                    vtkSmartPointer<vtkTransform> vtktransformation = vtkSmartPointer<vtkTransform>::New();
                    pcl::visualization::PCLVisualizer::convertToVtkMatrix(transformation, vtkmatrix);
                    vtktransformation->SetMatrix(&(*vtkmatrix));
                    viewer_->removeShape("PLYModel");
                    viewer_->addModelFromPLYFile (std::string("/home/sifi/Dropbox/mujin/lancable_terminal/lancable_terminal_fine_m.ply"),
                            vtktransformation);

                }
                else
                {
                    pcl::console::print_error ("Alignment failed!\n");
                    t = t*rotaffine;
                    vtkSmartPointer<vtkMatrix4x4> vtkmatrix = vtkSmartPointer<vtkMatrix4x4>::New();
                    vtkSmartPointer<vtkTransform> vtktransformation = vtkSmartPointer<vtkTransform>::New();
                    for (size_t row = 0; row < 4; row++) {
                        for (size_t col = 0; col < 4; col++) {
                            vtkmatrix->SetElement(row,col,t(row,col));
                        }
                    }
                    //pcl::visualization::PCLVisualizer::convertToVtkMatrix(t, vtkmatrix);
                    vtktransformation->SetMatrix(&(*vtkmatrix));
                    viewer_->removeShape("PLYModel");
                    viewer_->addModelFromPLYFile (std::string("/home/sifi/Dropbox/mujin/lancable_terminal/lancable_terminal_fine_m.ply"), vtktransformation);
                    viewer_->spinOnce();
                    boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
                }
            }
        */
            /*}}}*/
            ///////////////////////////////////////

            pcl::console::print_highlight ("Starting terminal alignment...\n");
            pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
            size_t rotnum = 4;
            for (size_t irot = 0; irot < rotnum; irot++) {
                pcl::console::print_highlight ("Starting terminal alignment... (%d)\n", irot);

                Eigen::Affine3f rotaffine;
                rotaffine = Eigen::AngleAxisf(2.0*M_PI/rotnum,v);
                Eigen::Affine3f tnew;
                tnew = t;
                for (size_t i = 0; i < irot; i++) {
                    tnew = tnew * rotaffine;
                }

                viewer_->removeCoordinateSystem();
                viewer_->addCoordinateSystem(0.1, tnew);
                PointCloudInputPtr transformedterminalcloud(new PointCloudInput);
                pcl::copyPointCloud<PointNT>(*voxelterminalcloud, *transformedterminalcloud);
                pcl::transformPointCloudWithNormals (*transformedterminalcloud, *transformedterminalcloud, tnew); //TODO incorrect

                align.setInputSource (transformedterminalcloud); //(voxelterminalcloud);
                align.setSourceFeatures (terminal_features);
                align.setInputTarget (firstterminalscenepoints);
                align.setTargetFeatures (scene_features);
                align.setMaximumIterations (10000); // Number of RANSAC iterations
                align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
                align.setCorrespondenceRandomness (2); // Number of nearest features to use
                align.setSimilarityThreshold (0.9f); // Polygonal edge length similarity threshold
                align.setMaxCorrespondenceDistance(1.5f * leafsize); // Inlier threshold
                align.setInlierFraction (0.4f);//(0.25f); // Required inlier fraction for accepting a pose hypothesis
                PointCloudInputPtr object_aligned(new PointCloudInput());
                {
                    pcl::ScopeTime t("Alignment");
                    align.align (*object_aligned);
                }

                if (align.hasConverged ())
                {
                    // Print results
                    printf ("\n");
                    Eigen::Matrix4f transformation = align.getFinalTransformation ();
                    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
                    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
                    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
                    pcl::console::print_info ("\n");
                    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
                    pcl::console::print_info ("\n");
                    pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), voxelterminalcloud->size ());

                    // Show alignment
                    //PointCloudInputPtr terminalcloudforvis(new PointCloudInput());
                    //pcl::visualization::PointCloudColorHandlerRGBField<PointNT> rgbfield(terminalcloud_);
                    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBNormal> rgbfield(extractedpoints, 0, 255, 0);
                    //pcl::copyPointCloud(terminalcloud_, terminalcloudforvis);

                    //pcl::transformPointCloudWithNormals (*terminalcloud_, *terminalcloudforvis, transformation);
                    //pcl::visualization::PointCloudColorHandlerCustom<PointNT> rgbfield(object_aligned, 0, 255, 0);
                    //viewer_->addPointCloud<PointNT> (terminalcloudforvis, rgbfield, "firstterminal");
                    ////viewer_->addPointCloud<PointNT> (object_aligned, rgbfield, "firstterminal");
                    //viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "firstterminal");

                    vtkSmartPointer<vtkMatrix4x4> vtkmatrix = vtkSmartPointer<vtkMatrix4x4>::New();
                    vtkSmartPointer<vtkTransform> vtktransformation = vtkSmartPointer<vtkTransform>::New();
                    pcl::visualization::PCLVisualizer::convertToVtkMatrix(tnew.inverse()*transformation, vtkmatrix);
                    vtktransformation->SetMatrix(&(*vtkmatrix));
                    viewer_->removeShape("PLYModel");
                    viewer_->addModelFromPLYFile (std::string("/home/sifi/Dropbox/mujin/lancable_terminal/lancable_terminal_fine_m.ply"),
                            vtktransformation);
                    break;
                }
                else
                {
                    pcl::console::print_error ("Alignment failed!\n");
                    vtkSmartPointer<vtkMatrix4x4> vtkmatrix = vtkSmartPointer<vtkMatrix4x4>::New();
                    vtkSmartPointer<vtkTransform> vtktransformation = vtkSmartPointer<vtkTransform>::New();
                    for (size_t row = 0; row < 4; row++) {
                        for (size_t col = 0; col < 4; col++) {
                            vtkmatrix->SetElement(row,col,tnew(row,col));
                        }
                    }
                    vtktransformation->SetMatrix(&(*vtkmatrix));
                    viewer_->removeShape("PLYModel");
                    viewer_->addModelFromPLYFile (std::string("/home/sifi/Dropbox/mujin/lancable_terminal/lancable_terminal_fine_m.ply"), vtktransformation);
                    //pcl::visualization::PointCloudColorHandlerCustom<PointNT> rgbfield(object_aligned, 0, 255, 0);
                    //viewer_->addPointCloud<PointNT> (terminalcloudforvis, rgbfield, "firstterminal");
                    ////viewer_->addPointCloud<PointNT> (object_aligned, rgbfield, "firstterminal");
                    //viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "firstterminal");

                    viewer_->spinOnce();
                    //boost::this_thread::sleep(boost::posix_time::milliseconds(1000));

                }
            }
        }

    }

    /* 
     * param[in] terminalcylindercoeffs: terminalcylindercoeffs
     * param[out] indices: scene point indices inside terminal cylinder
     * return: the size of indices
     */
    size_t findScenePointIndicesInsideCylinder(pcl::ModelCoefficients::Ptr terminalcylindercoeffs, pcl::PointIndices::Ptr indices)/*{{{*/
    {
        Eigen::Vector3f w(terminalcylindercoeffs->values[3], terminalcylindercoeffs->values[4], terminalcylindercoeffs->values[5]);
        w.normalize(); // just in case

        double radius      = terminalcylindercoeffs->values[6];
        double upperheight = terminalcylindercoeffs->values[7];
        double lowerheight = terminalcylindercoeffs->values[8];
        size_t count = 0;

        for (size_t i = 0; i < input_->points.size(); i++) {
            Eigen::Vector3f v(input_->points[i].x - terminalcylindercoeffs->values[0],
                    input_->points[i].y - terminalcylindercoeffs->values[1],
                    input_->points[i].z - terminalcylindercoeffs->values[2]);
            Eigen::Vector3f projectedv = v.dot(w) * w;
            double h = projectedv.norm();
            if (h > upperheight || h < lowerheight) {
                continue;
            }
            Eigen::Vector3f radiationv = v - projectedv;
            double r2 = radiationv.norm();
            if (r2 > radius) {
                continue;
            }
            indices->indices.push_back(i);
            count++;
        }
        return count;
    }/*}}}*/

    pcl::PointIndices::Ptr findClosePointsIndices(pcl::PointXYZ pt, double radius = 0) { /*{{{*/
        if (radius == 0) {
            radius = cableradius_*2;
        }
        pcl::PointIndices::Ptr k_indices(new pcl::PointIndices());
        std::vector<float> k_sqr_distances;
        kdtree_->setInputCloud(points_);
        kdtree_->radiusSearch (pt, radius, k_indices->indices, k_sqr_distances);
        return k_indices;
    } /*}}}*/

    bool estimateCylinderAroundPointsIndices (pcl::PointIndices::Ptr pointsindices, CableSlice& slice, pcl::PointXYZ centerpt = pcl::PointXYZ(), const Eigen::Vector3f& initialaxis = Eigen::Vector3f(), double eps_angle=0.0) /*{{{*/
    {
        // Create the segmentation object
        pcl::SACSegmentationFromNormals<PointNT, PointNT> seg;
        pcl::PointIndices::Ptr cylinderinlierindices(new pcl::PointIndices());
        // Optional
        seg.setOptimizeCoefficients (true);
        Eigen::Vector3f axis;
        axis = initialaxis;
        axis.normalize();
        seg.setAxis (axis);
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

        if (slice.cylinderindices->indices.size() == 0) {
            return false;
        }

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
        return validateCableSlice(slice);
    } /*}}}*/

    bool validateCableSlice (CableSlice& slice) /*{{{*/
    {
        PCL_INFO("[validateCableSlice] radius: %f, given: %f\n", slice.cylindercoeffs->values[6], cableradius_);
        if(cableradius_* 0.65 > slice.cylindercoeffs->values[6] || slice.cylindercoeffs->values[6] > cableradius_* 1.35 ) {
            return false;
        }
        return true;
    } /*}}}*/

    void keyboard_callback (const pcl::visualization::KeyboardEvent& event) /*{{{*/
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
    PointCloudInputPtr terminalcloud_;
    pcl::ModelCoefficients::Ptr terminalcylindercoeffs_;
    /* note: terminalcylindermodelcoeffs has '''9''' values
     * 0, 1, 2: pos
     * 3, 4, 5: dir
     * 6      : radius
     * 7,8    : upperheight, lowerheight
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_;
    pcl::PointIndices::Ptr indices_;
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_;

    double distthreshold_cylindermodel_;
    double scenesampling_radius_;
    double cableradius_;
    double cableslicelen_;
    std::vector<Cable> cables_;
    const int tryfindingpointscounts_;

    boost::shared_ptr<boost::thread> viewerthread_;
    boost::mutex viewer_mutex_;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;

};


} // namespace pcl_cable_detection
#endif /* end of include guard */

