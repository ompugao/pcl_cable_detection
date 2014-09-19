#ifndef __PCL_CABLE_DETECTION__
#define __PCL_CABLE_DETECTION__

#include <boost/random.hpp>
#include <boost/array.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/io.h>
#include <pcl/common/time.h>
#include <pcl/common/pca.h>
#include <pcl/common/intersections.h>
#include <pcl/console/print.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/pcl_search.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/visualization/cloud_viewer.h>
#include <cv.hpp>
#include <algorithm> //copy,fill
#include <iterator> //back_inserter
#include <cstdlib> //random
#include <cmath>
#include <vtkTransform.h>

template<class T>
bool pairCompare1(const std::pair<T, T> & x, const std::pair<T, T> & y) {
  return x.first < y.first; 
}
template<class T>
bool pairCompare2(const std::pair<T, T> & x, const std::pair<T, T> & y) {
  return x.second < y.second; 
}

/// \brief Return the minimal quaternion that orients sourcedir to targetdir
///
/// \ingroup affine_math
/// \param sourcedir direction of the original vector, 3 values
/// \param targetdir new direction, 3 values
/// copy from openrave, quatRotateDirection function
Eigen::Affine3f AffineFromRotateDirection(Eigen::Vector3f& sourcedir, Eigen::Vector3f& targetdir)/*{{{*/
{
    Eigen::Affine3f affinetransform;

    Eigen::Vector3f rottodirection = sourcedir.cross(targetdir);
    float fsin = rottodirection.norm();
    float fcos = rottodirection.dot(targetdir);
    Eigen::Vector3f torient;
    if( fsin > 0 ) {
        //return quatFromAxisAngle(rottodirection*(1/fsin), MATH_ATAN2(fsin, fcos));
        affinetransform = Eigen::AngleAxisf(atan2(fsin, fcos), rottodirection*(1/fsin));
        return affinetransform;
    }
    if( fcos < 0 ) {
        // hand is flipped 180, rotate around x axis
        rottodirection[0] = 1;
        rottodirection[1] = 0;
        rottodirection[2] = 0;

        rottodirection -= sourcedir * sourcedir.dot(rottodirection);
        if( rottodirection.squaredNorm() < 1e-8 ) {
            rottodirection[0] = 0; rottodirection[1] = 0; rottodirection[2] = 1;
            rottodirection -= sourcedir * sourcedir.dot(rottodirection);
        }
        rottodirection.norm();
        affinetransform = Eigen::AngleAxisf(atan2(fsin, fcos), rottodirection);
        return affinetransform;
    }
    return Eigen::Affine3f::Identity();
}/*}}}*/

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

template<typename PointT>
size_t FindPointIndicesInsideCylinder(const typename pcl::PointCloud<PointT>::Ptr cloud, pcl::ModelCoefficients::Ptr terminalcylindercoeffs, pcl::PointIndices::Ptr indices)/*{{{*/
{
    Eigen::Vector3f w(terminalcylindercoeffs->values[3], terminalcylindercoeffs->values[4], terminalcylindercoeffs->values[5]);
    w.normalize(); // just in case

    double radius      = terminalcylindercoeffs->values[6];
    double upperheight = terminalcylindercoeffs->values[7];
    double lowerheight = terminalcylindercoeffs->values[8];
    size_t count = 0;

    for (size_t i = 0; i < cloud->points.size(); i++) {
        Eigen::Vector3f v(cloud->points[i].x - terminalcylindercoeffs->values[0],
                cloud->points[i].y - terminalcylindercoeffs->values[1],
                cloud->points[i].z - terminalcylindercoeffs->values[2]);
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

template<typename PointT>
void computeTipPointByProjectIntoLine(const typename pcl::PointCloud<PointT>::Ptr cloud, const Eigen::Vector3f& line_dir, const Eigen::Vector3f& line_pt, float& maxtiplength, Eigen::Vector3f& maxtippoint, Eigen::Vector3f& maxtippointonline, float& mintiplength, Eigen::Vector3f& mintippoint, Eigen::Vector3f& mintippointonline)/*{{{*/
{
    maxtiplength = -std::numeric_limits<float>::max(); //std::numeric_limits<float>::lowest();
    mintiplength = std::numeric_limits<float>::max(); //std::numeric_limits<float>::lowest();
    //Eigen::Vector4f line_pt(line[0], line[1], line[2], 0);
    //Eigen::Vector4f line_dir(line[3], line[4], line[5], 0);
    for (size_t ilinepoint = 0; ilinepoint < cloud->points.size(); ilinepoint++) {
        Eigen::Vector3f pt(cloud->points[ilinepoint].x, cloud->points[ilinepoint].y, cloud->points[ilinepoint].z);
        // double k = (DOT_PROD_3D (points[i], p21) - dotA_B) / dotB_B;
        float k = (pt.dot (line_dir) - line_pt.dot (line_dir)) / line_dir.dot (line_dir);
        if (maxtiplength < k) {
            maxtiplength = k;
            //Eigen::Vector4f pp = line_pt + k * line_dir;
            maxtippointonline = line_pt + k * line_dir;
            maxtippoint = pt;
            //tippoint = pp.head<3>();
        }
        if (mintiplength > k) {
            mintiplength = k;
            mintippointonline = line_pt + k * line_dir;
            mintippoint = pt;
        }
        //std::cout << "k: " << k << " maxtiplength: " << maxtiplength<<" mintiplength: " << mintiplength<< std::endl;
        //std::cout << "    maxtippoint: " << maxtippoint[0] << ", " << maxtippoint[1] << ", " << maxtippoint[2] << std::endl;
        //std::cout << "    mintippoint: " << mintippoint[0] << ", " << mintippoint[1] << ", " << mintippoint[2] << std::endl;
    }
}/*}}}*/

// http://www.pcl-users.org/concave-hulls-td3802830.html
template<typename PointT>
double computeMeshArea(pcl::PointCloud<PointT> &cloud, std::vector<pcl::Vertices> mesh)/*{{{*/
{
    double surface_area = 0;
    float x[3];
    float y[3];
    float z[3];
    float d[3];
    double s;
    for(int i = 0; i < mesh.size(); i++)
    {
        x[0] = cloud[mesh[i].vertices[1]].x - cloud[mesh[i].vertices[0]].x;
        x[1] = cloud[mesh[i].vertices[2]].x - cloud[mesh[i].vertices[1]].x;
        x[2] = cloud[mesh[i].vertices[0]].x - cloud[mesh[i].vertices[2]].x;
        y[0] = cloud[mesh[i].vertices[1]].y - cloud[mesh[i].vertices[0]].y;
        y[1] = cloud[mesh[i].vertices[2]].y - cloud[mesh[i].vertices[1]].y;
        y[2] = cloud[mesh[i].vertices[0]].y - cloud[mesh[i].vertices[2]].y;
        z[0] = cloud[mesh[i].vertices[1]].z - cloud[mesh[i].vertices[0]].z;
        z[1] = cloud[mesh[i].vertices[2]].z - cloud[mesh[i].vertices[1]].z;
        z[2] = cloud[mesh[i].vertices[0]].z - cloud[mesh[i].vertices[2]].z;
        d[0] = sqrt(x[0]*x[0]+y[0]*y[0]+z[0]*z[0]);
        d[1] = sqrt(x[1]*x[1]+y[1]*y[1]+z[1]*z[1]);
        d[2] = sqrt(x[2]*x[2]+y[2]*y[2]+z[2]*z[2]);
        s = (d[0]+d[1]+d[2])/2.;
        //std::cout << sqrt(s*(s-d[0])*(s-d[1])*(s-d[2])) <<'\t';
        surface_area += sqrt(s*(s-d[0])*(s-d[1])*(s-d[2]));
    }
    std::cout << surface_area << std::endl;
}/*}}}*/

template<typename Point>
std::vector<Eigen::Vector3f> minAreaRect(pcl::ModelCoefficients::Ptr planecoefficients, const typename pcl::PointCloud<Point>::ConstPtr &projected_cloud, Eigen::Vector3f v = Eigen::Vector3f::Zero())/*{{{*/
{
    std::vector<Eigen::Vector3f> table_top_bbx;

    // Project points onto the table plane
    //pcl::ProjectInliers<Point> proj;
    //proj.setModelType(pcl::SACMODEL_PLANE);
    //pcl::PointCloud<Point> projected_cloud;
    //proj.setInputCloud(cloud);
    //proj.setModelCoefficients(table_coefficients_const_);
    //proj.filter(projected_cloud);

    // store the table top plane parameters
    Eigen::Vector3f plane_normal;
    plane_normal.x() = planecoefficients->values[0];
    plane_normal.y() = planecoefficients->values[1];
    plane_normal.z() = planecoefficients->values[2];
    // compute an orthogonal normal to the plane normal
    if (v == Eigen::Vector3f::Zero()) {
        v = plane_normal.unitOrthogonal();
    }
    // take the cross product of the two normals to get
    // a thirds normal, on the plane
    Eigen::Vector3f u = plane_normal.cross(v);

    // project the 3D point onto a 2D plane
    std::vector<cv::Point2f> points;
    // choose a point on the plane
    Eigen::Vector3f p0(projected_cloud->points[0].x,
            projected_cloud->points[0].y,
            projected_cloud->points[0].z);
    for(unsigned int ii=0; ii<projected_cloud->points.size(); ii++)
    {
        Eigen::Vector3f p3d(projected_cloud->points[ii].x,
                projected_cloud->points[ii].y,
                projected_cloud->points[ii].z);

        // subtract all 3D points with a point in the plane
        // this will move the origin of the 3D coordinate system
        // onto the plane
        p3d = p3d - p0;

        cv::Point2f p2d;
        p2d.x = p3d.dot(u);
        p2d.y = p3d.dot(v);
        points.push_back(p2d);
    }

    cv::Mat points_mat(points);
    cv::RotatedRect rrect = cv::minAreaRect(points_mat);
    cv::Point2f rrPts[4];
    rrect.points(rrPts);

    //store the table top bounding points in a vector
    for(unsigned int ii=0; ii<4; ii++)
    {
        Eigen::Vector3f pbbx(rrPts[ii].x*u + rrPts[ii].y*v + p0);
        table_top_bbx.push_back(pbbx);
    }
    Eigen::Vector3f center(rrect.center.x*u + rrect.center.y*v + p0);
    table_top_bbx.push_back(center);

    return table_top_bbx;
}/*}}}*/

/** \brief compute curvature histogram for each point using a given radius
 */
template<typename PointT, int N>
void computeCurvatureHistogram(const typename pcl::PointCloud<PointT>::Ptr cloud,/*{{{*/
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
}/*}}}*/

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
    typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerNT;
    typedef pcl::FPFHSignature33 FeatureT;
    typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
    typedef pcl::PointCloud<FeatureT> FeatureCloudT;

    CableDetection(PointCloudInputPtr terminalcloud, Eigen::Vector3f terminalaxis)
        : tryfindingpointscounts_(1)
    {
        cableradius_ = 0;
        distthreshold_cylindermodel_ = 0;
        scenesampling_radius_ = 0;

        // compute the bounding cylinder of the terminal
        terminalcloud_ = terminalcloud;
        terminalcylindercoeffs_.reset(new pcl::ModelCoefficients());
        _computeBoundingCylinder(terminalcloud_, terminalaxis, terminalcylindercoeffs_);
        srand (time(NULL));
        selected_point_index = 0;
    }
    // setter
    /*{{{*/
    void setInputCloud(PointCloudInputPtr input)
    {
        input_ = input;
        if (!!viewer_) {
            boost::mutex::scoped_lock lock(viewer_mutex_);
            //viewer_->removePointCloud("inputcloud");
            viewer_->removeAllPointClouds();
            viewer_->removeAllShapes();
            bool coordinatesystemremoved = true;
            while(coordinatesystemremoved) {
                coordinatesystemremoved = viewer_->removeCoordinateSystem();
            }
            viewer_->addCoordinateSystem(0.5);
            viewer_->addPointCloud<PointNT> (input_, "inputcloud");
            //viewer_->addPointCloudNormals<PointNT> (input_, 1, 0.001, "inputcloud");
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
        cableslicelen_ = 2 * cableradius_;
    }
    void setThresholdCylinderModel(double distthreshold_cylindermodel) {
        distthreshold_cylindermodel_ = distthreshold_cylindermodel;
    }
    void setSceneSamplingRadius(double scenesampling_radius) {
        scenesampling_radius_ = scenesampling_radius;
    }
    void setTerminalCenterCloud(pcl::PointCloud<pcl::PointWithRange>::Ptr cloudcenters)
    {
        cloudcenters_ = cloudcenters;
    }
/*}}}*/

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
            //viewer_->registerAreaPickingCallback(boost::bind(&CableDetection::area_picking_callback, this,_1));
            viewer_->registerPointPickingCallback(boost::bind(&CableDetection::point_picking_callback2, this, _1));
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
    void _computeBoundingCylinder(PointCloudInputPtr cloud, Eigen::Vector3f axis, pcl::ModelCoefficients::Ptr cylindercoeffs) { /*{{{*/
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
        std::vector<Cable> dummycables;
        std::vector<EndPointInfo> endpointinfos;
        Cable cable = findCableFromPoint(selectedpoint, dummycables, endpointinfos);
        // do not lock viewer mutex!!! the function who calls point_picking_callback itself is locking mutex
        viewer_->removeAllShapes();
        visualizeCable(cable);
        //findCableTerminal(cable, 0.027);
        findCableTerminal(cable, 0.018);
    } /*}}}*/
    void point_picking_callback2 (const pcl::visualization::PointPickingEvent& event) /*{{{*/
    {
        //pcl::PointXYZ pt;
        //event.getPoint (pt.x, pt.y, pt.z);
        size_t idx = event.getPointIndex ();
        std::cout << "picking point index: " << idx << std::endl;

        selected_points[selected_point_index].x = input_->points[idx].x;
        selected_points[selected_point_index].y = input_->points[idx].y;
        selected_points[selected_point_index].z = input_->points[idx].z;
        if (selected_point_index == selected_points.size() - 1)
        {
            pcl::console::setVerbosityLevel (pcl::console::L_DEBUG);
            std::cout << selected_points[0] << std::endl;
            std::cout << selected_points[1] << std::endl;
            Eigen::Vector3f dir(selected_points[1].x - selected_points[0].x,
                    selected_points[1].y - selected_points[0].y,
                    selected_points[1].z - selected_points[0].z);
            dir.normalize();
            Cable cable;
            Eigen::Affine3f terminaltransform;
            trackCableSimple(selected_points[0], dir, cable, terminaltransform);
/*{{{*/
            /*
            pcl::console::print_highlight("extract neareset points...\n");
            std::vector<int> k_indices;
            std::vector<float> k_sqr_distances;
            kdtree_->radiusSearch (selected_points[selected_point_index], 0.01, k_indices, k_sqr_distances);
            pcl::PointIndices::Ptr indices(new pcl::PointIndices);
            indices->indices.swap(k_indices);
            pcl::ExtractIndices<PointNT> extract;
            PointCloudInputPtr terminalscenepoints(new PointCloudInput());
            extract.setInputCloud (input_);
            extract.setIndices (indices);
            //extract.setNegative (true);
            extract.filter (*terminalscenepoints);
            std::cout << "points size: " << terminalscenepoints->points.size() << ", indices size: " << indices->indices.size() << std::endl;
            viewer_->removePointCloud("extracted points to fit plane");
            viewer_->addPointCloud<PointNT> (terminalscenepoints, pcl::visualization::PointCloudColorHandlerCustom<PointNT> (terminalscenepoints, 255, 128, 255),  "extracted points to fit plane");
            viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "extracted points to fit plane");

            pcl::PointIndices::Ptr inlierindices(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr planemodelcoeffs(new pcl::ModelCoefficients);
            double planethreshold = 0.0003; //0.0007
            pcl::console::print_highlight("fit plane...\n");
            // Create the segmentation object
            pcl::SACSegmentation<PointNT> seg;
            // Optional
            seg.setOptimizeCoefficients (true);
            //seg.setAxis(...);
            // Mandatory
            seg.setModelType (pcl::SACMODEL_PLANE);
            seg.setMethodType (pcl::SAC_RANSAC);
            //seg.setMethodType (pcl::SAC_RRANSAC);
            seg.setMaxIterations(50000);
            seg.setDistanceThreshold (planethreshold);
            seg.setInputCloud (terminalscenepoints);
            seg.segment (*inlierindices, *planemodelcoeffs);

            if (inlierindices->indices.size () == 0)
            {
                pcl::console::print_highlight("failed to fit plane...\n");
                goto finally;
            }
            PointCloudInputPtr pointsonplane(new PointCloudInput());
            extract.setInputCloud (terminalscenepoints);
            extract.setIndices (inlierindices);
            //extract.setNegative (true);
            extract.filter (*pointsonplane);
            viewer_->removePointCloud("pointsonplane");
            viewer_->addPointCloud<PointNT> (pointsonplane, pcl::visualization::PointCloudColorHandlerCustom<PointNT> (pointsonplane, 0, 255, 0),  "pointsonplane");
            viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "pointsonplane");

            viewer_->removeShape("plane");
            viewer_->addPlane(*planemodelcoeffs, selected_points[selected_point_index].x, selected_points[selected_point_index].y, selected_points[selected_point_index].z);
            *//*}}}*/
        }
finally:
        selected_point_index = (selected_point_index+1)%selected_points.size();
    } /*}}}*/
    boost::array<pcl::PointXYZ,2> selected_points;
    size_t selected_point_index;

    bool trackCableSimple(pcl::PointXYZ initialpt, Eigen::Vector3f dir, Cable& cable, Eigen::Affine3f& terminaltransform, std::string terminalindex="")
    {
        CableSlicePtr slice, oldslice, baseslice;
        dir.normalize();
        pcl::PointIndices::Ptr k_indices;
        pcl::PointXYZ searchpoint;
        searchpoint = initialpt;
        while(true) {
            slice.reset(new CableSlice());
            k_indices = _findClosePointsIndices(searchpoint, cableradius_*2);
            if (k_indices->indices.size() == 0) {
                PCL_DEBUG("endpoint?\n");
                break;
            }
            Eigen::Vector4d centroid;
            pcl::compute3DCentroid (*input_, *k_indices, centroid);

            slice->searchedindices->indices.swap(k_indices->indices);
            slice->centerpt_ = searchpoint;
            //slice->centerpt_.x = centroid[0];
            //slice->centerpt_.y = centroid[1];
            //slice->centerpt_.z = centroid[2];
            slice->cylindercoeffs->values.resize(9);
            slice->cylindercoeffs->values[0] = centroid[0];
            slice->cylindercoeffs->values[1] = centroid[1];
            slice->cylindercoeffs->values[2] = centroid[2]+cableradius_;
            slice->cylindercoeffs->values[6] = cableradius_;
            if (!oldslice) {
                //Eigen::Vector3f slicedir(oldslice->cylindercoeffs->values[0] - slice->cylindercoeffs->values[0],
                        //oldslice->cylindercoeffs->values[1] - slice->cylindercoeffs->values[1],
                        //oldslice->cylindercoeffs->values[2] - slice->cylindercoeffs->values[2]);
                //slicedir.normalize();
                slice->cylindercoeffs->values[3] = dir[0];
                slice->cylindercoeffs->values[4] = dir[1];
                slice->cylindercoeffs->values[5] = dir[2];
                searchpoint.x = centroid[0] + dir[0] * cableradius_*4;
                searchpoint.y = centroid[1] + dir[1] * cableradius_*4;
                searchpoint.z = centroid[2] + dir[2] * cableradius_*4;
            } else {
                Eigen::Vector3f slicedir(slice->cylindercoeffs->values[0] - oldslice->cylindercoeffs->values[0],
                        slice->cylindercoeffs->values[1] - oldslice->cylindercoeffs->values[1],
                        slice->cylindercoeffs->values[2] - oldslice->cylindercoeffs->values[2]);
                slicedir.normalize();
                slice->cylindercoeffs->values[3] = slicedir[0];
                slice->cylindercoeffs->values[4] = slicedir[1];
                slice->cylindercoeffs->values[5] = slicedir[2];
                searchpoint.x = centroid[0] + slicedir[0] * cableradius_*4;
                searchpoint.y = centroid[1] + slicedir[1] * cableradius_*4;
                searchpoint.z = centroid[2] + slicedir[2] * cableradius_*4;
            }
            oldslice = slice;
            cable.push_front(slice);
            std::cout << "push_front slice" << std::endl;
        }
        
        //Eigen::Affine3f transform = AffineFromRotateDirection(Eigen::Vector3f(0,0,1),
                //Eigen::Vector3f(cable.begin()->cylindercoeffs->values[3],
                    //cable.begin()->cylindercoeffs->values[4],
                    //cable.begin()->cylindercoeffs->values[5]));

        //pcl::PointXYZ pt;
        //pt.x = cable.begin()->cylindercoeffs->values[3];
        //pt.y = cable.begin()->cylindercoeffs->values[4];
        //pt.z = cable.begin()->cylindercoeffs->values[5];
        //k_indices = _findClosePointsIndices(pt, cableradius_*);

        if (cable.size() == 0)
        {
            PCL_WARN("could not find any cable!\n");
            return false;
        }
        pcl::ModelCoefficients::Ptr terminalcoeffs(new pcl::ModelCoefficients);
        std::copy(terminalcylindercoeffs_->values.begin(), terminalcylindercoeffs_->values.end(),
                std::back_inserter(terminalcoeffs->values));
        terminalcoeffs->values[0] = (*cable.begin())->cylindercoeffs->values[0];
        terminalcoeffs->values[1] = (*cable.begin())->cylindercoeffs->values[1];
        terminalcoeffs->values[2] = (*cable.begin())->cylindercoeffs->values[2];
        terminalcoeffs->values[3] = (*cable.begin())->cylindercoeffs->values[3];
        terminalcoeffs->values[4] = (*cable.begin())->cylindercoeffs->values[4];
        terminalcoeffs->values[5] = (*cable.begin())->cylindercoeffs->values[5];
        terminalcoeffs->values[6] += 0.010;
        terminalcoeffs->values[7] += 0.020;
        terminalcoeffs->values[8] -= 0.010;
        
        bool status = _estimateTerminalFromInitialCoeffs2(terminalcoeffs, terminaltransform, terminalindex, true);
        return status;
    }

    bool _estimateTerminalFromInitialCoeffs2(pcl::ModelCoefficients::Ptr terminalcoeffs, Eigen::Affine3f& detectedterminaltransform, std::string terminalindex = "", bool enablefancyvisualization=false)
    {
        double planethreshold               = 0.0005; //0.0007
        double extractinliers_distthreshold = 0.001;
        int    indicessize_protruding       = 140;//200
        int    indicessize_notprotruding    = 30;//200
        double z_offset = 0.0032;
        double y_offset = 0.00535;
        double x_offset = 0.0058;
        double flatheadoffset = 0.012; // 0.012 // 先端から平面が続く長さ
        double areasize_threshold = 100 * 1e-6;//135 * 1e-6; // > 6.4 * flatheadoffset && < 11.6 * flatheadoffset
        double radiusoutlierremovalsize = 0.0003;

        pcl::ExtractIndices<PointNT> extract;
        pcl::SACSegmentation<PointNT> seg;
        pcl::ConvexHull<PointNT> chull;
        pcl::ConcaveHull<PointNT> concavehull;
        pcl::ProjectInliers<PointNT> proj;

        // extract terminal scene pointcloud
        std::cout << "_findScenePointIndicesInsideCylinder terminal coeffs: " << *terminalcoeffs << std::endl;
        //viewer_->removeShape("cylinder"+terminalindex);
        //viewer_->addCylinder(*terminalcoeffs,"cylinder"+terminalindex);
        pcl::PointIndices::Ptr terminalscenepointsindices(new pcl::PointIndices);
        size_t points = _findScenePointIndicesInsideCylinder(terminalcoeffs, terminalscenepointsindices);
        std::cout << points << " points for terminal" << std::endl;

        PointCloudInputPtr terminalscenepoints(new PointCloudInput());
        extract.setInputCloud (input_);
        extract.setIndices (terminalscenepointsindices);
        extract.setNegative (false);
        extract.filter (*terminalscenepoints);
/*{{{*/
        if (enablefancyvisualization) {
            viewer_->removePointCloud("terminalscenepoints" + terminalindex);
            viewer_->addPointCloud<PointNT> (terminalscenepoints, pcl::visualization::PointCloudColorHandlerCustom<PointNT> (terminalscenepoints, 255, 128, 255),  "terminalscenepoints" + terminalindex);
            viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "terminalscenepoints" + terminalindex);
        }
/*}}}*/
        pcl::PointIndices::Ptr inlierindices1(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr planemodelcoeffs1(new pcl::ModelCoefficients);
        seg.setOptimizeCoefficients (true);
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        //seg.setMethodType (pcl::SAC_RRANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold (planethreshold);
        seg.setInputCloud (terminalscenepoints);
        seg.segment (*inlierindices1, *planemodelcoeffs1);

        if (inlierindices1->indices.size () == 0)
        {
            pcl::console::print_highlight("failed to fit plane...\n");
            return false;
        }
        Eigen::Vector4f lookatvector4, centroidpt4;
        pcl::compute3DCentroid (*terminalscenepoints, centroidpt4);
        lookatvector4 = centroidpt4.normalized();

        if (Eigen::Vector3f(planemodelcoeffs1->values[0], planemodelcoeffs1->values[1], planemodelcoeffs1->values[2]).dot(lookatvector4.segment<3>(0)) > 0) {  // if the normal faces to the z-axis of camera coordinates
            planemodelcoeffs1->values[0] *= -1; planemodelcoeffs1->values[1] *= -1; planemodelcoeffs1->values[2] *= -1; planemodelcoeffs1->values[3] *= -1;
        }
        std::cout << "plane1 coeffs: " << *planemodelcoeffs1 << std::endl;
        PointCloudInputPtr pointsonplane1(new PointCloudInput());
        extract.setInputCloud (terminalscenepoints);
        extract.setIndices (inlierindices1);
        extract.setNegative (false);
        extract.filter (*pointsonplane1);

        if (enablefancyvisualization) {/*{{{*/
            viewer_->removePointCloud("pointsonplane1_" + terminalindex);
            viewer_->addPointCloud<PointNT> (pointsonplane1, pcl::visualization::PointCloudColorHandlerCustom<PointNT> (pointsonplane1, 0, 255, 0),  "pointsonplane1_" + terminalindex);
            viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "pointsonplane1_" + terminalindex);

            viewer_->removeShape("plane1"+terminalindex);
            viewer_->addPlane(*planemodelcoeffs1, pointsonplane1->points[0].x,pointsonplane1->points[0].y,pointsonplane1->points[0].z, "plane1"+terminalindex);
        }
        /*}}}*/

        // Find the distance from point to plane.
        // http://mathworld.wolfram.com/Point-PlaneDistance.html
        pcl::PointIndices::Ptr inlierindices_enlarged1(new pcl::PointIndices);
        double denominator = sqrt(pow(planemodelcoeffs1->values[0], 2) + pow(planemodelcoeffs1->values[1], 2) + pow(planemodelcoeffs1->values[2], 2));
        for (size_t i = 0; i < terminalscenepoints->size(); i++) {
            double dist = terminalscenepoints->points[i].x * planemodelcoeffs1->values[0] + terminalscenepoints->points[i].y * planemodelcoeffs1->values[1] +  terminalscenepoints->points[i].z * planemodelcoeffs1->values[2] + planemodelcoeffs1->values[3];
            dist /= denominator;
            if (-extractinliers_distthreshold < dist && dist < extractinliers_distthreshold) {
                inlierindices_enlarged1->indices.push_back(i);
            }
        }
        std::cout << "inlierindices_enlarged1: " << inlierindices_enlarged1->indices.size() << std::endl;

        // recompute plane1
        pcl::console::print_highlight("recompute plane1\n");
        pcl::PointIndices::Ptr dummyinliers(new pcl::PointIndices);
        seg.setOptimizeCoefficients (true);
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold (planethreshold);
        seg.setInputCloud (terminalscenepoints);
        seg.setIndices (inlierindices_enlarged1);
        seg.segment (*dummyinliers, *planemodelcoeffs1);
        if (Eigen::Vector3f(planemodelcoeffs1->values[0], planemodelcoeffs1->values[1], planemodelcoeffs1->values[2]).dot(lookatvector4.segment<3>(0)) > 0) {  // if the normal faces to the z-axis of camera coordinates
            planemodelcoeffs1->values[0] *= -1; planemodelcoeffs1->values[1] *= -1; planemodelcoeffs1->values[2] *= -1; planemodelcoeffs1->values[3] *= -1;
        }

        // Find the distance from point to plane.
        // http://mathworld.wolfram.com/Point-PlaneDistance.html
        //pcl::PointIndices::Ptr inlierindices_enlarged1(new pcl::PointIndices);
        inlierindices_enlarged1->indices.resize(0);
        pcl::PointIndices::Ptr inlierindices_upper1(new pcl::PointIndices);
        pcl::PointIndices::Ptr inlierindices_lower1(new pcl::PointIndices);
        denominator = sqrt(pow(planemodelcoeffs1->values[0], 2) + pow(planemodelcoeffs1->values[1], 2) + pow(planemodelcoeffs1->values[2], 2));
        for (size_t i = 0; i < terminalscenepoints->size(); i++) {
            double dist = terminalscenepoints->points[i].x * planemodelcoeffs1->values[0] + terminalscenepoints->points[i].y * planemodelcoeffs1->values[1] +  terminalscenepoints->points[i].z * planemodelcoeffs1->values[2] + planemodelcoeffs1->values[3];
            dist /= denominator;
            if (-extractinliers_distthreshold < dist && dist < extractinliers_distthreshold) {
                inlierindices_enlarged1->indices.push_back(i);
            }
            else if (extractinliers_distthreshold < dist ) {
                inlierindices_upper1->indices.push_back(i);
            }
            else if (dist < - extractinliers_distthreshold) {
                inlierindices_lower1->indices.push_back(i);
            }
        }
        std::cout << "inlierindices_enlarged1: " << inlierindices_enlarged1->indices.size() << std::endl;
        std::cout << "inlierindices_upper1: " << inlierindices_upper1->indices.size() << std::endl;
        std::cout << "inlierindices_lower1: " << inlierindices_lower1->indices.size() << std::endl;

        PointCloudInputPtr terminalscenepoints2(new PointCloudInput());
        PointCloudInputPtr pointsonplane_enlarged1(new PointCloudInput());
        extract.setInputCloud (terminalscenepoints);
        extract.setIndices (inlierindices_enlarged1);
        extract.setNegative (false);
        extract.filter (*pointsonplane_enlarged1);
        extract.setNegative (true);
        extract.filter (*terminalscenepoints2);

        if (enablefancyvisualization) {/*{{{*/
            viewer_->removePointCloud("terminalscenepoints2_" + terminalindex);
            viewer_->addPointCloud<PointNT> (terminalscenepoints2, pcl::visualization::PointCloudColorHandlerCustom<PointNT> (terminalscenepoints2, 128, 255, 255),  "terminalscenepoints2_" + terminalindex);
            viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "terminalscenepoints2_" + terminalindex);
        }
        /*}}}*/

        pcl::PointIndices::Ptr inlierindices2(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr planemodelcoeffs2(new pcl::ModelCoefficients);
        planemodelcoeffs2->values.resize(4);
        pcl::console::print_highlight("fit plane 2...\n");
        seg.setOptimizeCoefficients (true);
        //seg.setAxis(...);
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        //seg.setMethodType (pcl::SAC_RRANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold (planethreshold);
        seg.setInputCloud (terminalscenepoints2);
        seg.segment (*inlierindices2, *planemodelcoeffs2);

        pcl::PointIndices::Ptr inlierindices_enlarged2(new pcl::PointIndices);
        pcl::PointIndices::Ptr inlierindices_upper2(new pcl::PointIndices);
        pcl::PointIndices::Ptr inlierindices_lower2(new pcl::PointIndices);
        PointCloudInputPtr pointsonplane2(new PointCloudInput());
        if (inlierindices2->indices.size () == 0) {
            pcl::console::print_highlight("failed to fit plane2\n");
        } else {
            if (Eigen::Vector3f(planemodelcoeffs2->values[0], planemodelcoeffs2->values[1], planemodelcoeffs2->values[2]).dot(lookatvector4.segment<3>(0)) > 0) {  // if the normal faces to the z-axis of camera coordinates
                planemodelcoeffs2->values[0] *= -1; planemodelcoeffs2->values[1] *= -1; planemodelcoeffs2->values[2] *= -1; planemodelcoeffs2->values[3] *= -1;
            }
            std::cout << "plane2 coeffs: " << *planemodelcoeffs2 << std::endl;
            double denominator = sqrt(pow(planemodelcoeffs2->values[0], 2) + pow(planemodelcoeffs2->values[1], 2) + pow(planemodelcoeffs2->values[2], 2));
            for (size_t i = 0; i < terminalscenepoints2->size(); i++) {
                double dist = terminalscenepoints2->points[i].x * planemodelcoeffs2->values[0] + terminalscenepoints2->points[i].y * planemodelcoeffs2->values[1] +  terminalscenepoints2->points[i].z * planemodelcoeffs2->values[2] + planemodelcoeffs2->values[3];
                dist /= denominator;
                if (-extractinliers_distthreshold < dist && dist < extractinliers_distthreshold) {
                    inlierindices_enlarged2->indices.push_back(i);
                }
                else if (extractinliers_distthreshold < dist ) {
                    inlierindices_upper2->indices.push_back(i);
                }
                else if (dist < - extractinliers_distthreshold) {
                    inlierindices_lower2->indices.push_back(i);
                }
            }
            std::cout << "inlierindices_enlarged2: " << inlierindices_enlarged2->indices.size() << std::endl;
            std::cout << "inlierindices_upper2: " << inlierindices_upper2->indices.size() << std::endl;
            std::cout << "inlierindices_lower2: " << inlierindices_lower2->indices.size() << std::endl;
            extract.setInputCloud (terminalscenepoints2);
            extract.setIndices (inlierindices2);
            extract.setNegative (false);
            extract.filter (*pointsonplane2);
            if (enablefancyvisualization) {
                viewer_->removePointCloud("pointsonplane2"+terminalindex);
                viewer_->addPointCloud<PointNT> (pointsonplane2, pcl::visualization::PointCloudColorHandlerCustom<PointNT> (pointsonplane2, 0, 0, 255),  "pointsonplane2"+terminalindex);
                viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "pointsonplane2"+terminalindex);

                viewer_->removeShape("plane2"+terminalindex);
                viewer_->addPlane(*planemodelcoeffs2, pointsonplane2->points[0].x,pointsonplane2->points[0].y,pointsonplane2->points[0].z, "plane2" + terminalindex);
            }

        }

        ////// estimate terminal pose
        Eigen::Affine3f terminaltransform = Eigen::Affine3f::Identity();
        Eigen::Vector3f dir(terminalcoeffs->values[3], terminalcoeffs->values[4], terminalcoeffs->values[5]);
        Eigen::Vector3f n1(planemodelcoeffs1->values[0], planemodelcoeffs1->values[1], planemodelcoeffs1->values[2]); //plane1.head<3>()
        Eigen::Vector3f n2(planemodelcoeffs2->values[0], planemodelcoeffs2->values[1], planemodelcoeffs2->values[2]);
        Eigen::Vector4f plane1(planemodelcoeffs1->values[0], planemodelcoeffs1->values[1], planemodelcoeffs1->values[2], planemodelcoeffs1->values[3]);
        Eigen::Vector4f plane2(planemodelcoeffs2->values[0], planemodelcoeffs2->values[1], planemodelcoeffs2->values[2], planemodelcoeffs2->values[3]);

        if ((inlierindices2->indices.size () != 0 && n1.dot(n2) < 0.4 && dir.dot(n1) < 0.4 && dir.dot(n1) < 0.4)
                && plane2.dot(centroidpt4) > 0 // < 0 なら裏が見えてる
                ) {
            // if it can estimate both planes
            Eigen::VectorXf line;
            pcl::planeWithPlaneIntersection(plane1, plane2, line, 0.05); //line[0] ~ line[2]: point //line[3] ~ line[5]: direction
            if (line.segment(3,3).dot(dir) < 0) {
                line[3] *= -1; line[4] *= -1; line[5] *= -1;
            }

            //pcl::ModelCoefficients line_coeff;/*{{{*/
            //line_coeff.values.resize(6);
            //for (size_t imodel = 0; imodel < 6; imodel++) {
                //line_coeff.values[imodel] = line[imodel];
            //}
            ////viewer_->addLine (line_coeff, "line"+terminalindex);/*}}}*/
            float maxtiplength = -std::numeric_limits<float>::max(); //std::numeric_limits<float>::lowest();
            Eigen::Vector3f tippoint;
            Eigen::Vector4f line_pt(line[0], line[1], line[2], 0);
            Eigen::Vector4f line_dir(line[3], line[4], line[5], 0);
            for (size_t ilinepoint = 0; ilinepoint < pointsonplane1->points.size(); ilinepoint++) {
                Eigen::Vector4f pt(pointsonplane1->points[ilinepoint].x, pointsonplane1->points[ilinepoint].y, pointsonplane1->points[ilinepoint].z, 0);
                // double k = (DOT_PROD_3D (points[i], p21) - dotA_B) / dotB_B;
                float k = (pt.dot (line_dir) - line_pt.dot (line_dir)) / line_dir.dot (line_dir);
                if (maxtiplength < k) {
                    maxtiplength = k;
                    Eigen::Vector4f pp = line_pt + k * line_dir;
                    tippoint = pp.head<3>();
                }
                //std::cout << "k: " << k << std::endl;
                //std::cout << "matxtiplength: " << maxtiplength << std::endl;
                //std::cout << "tippoint: " << tippoint[0] << ", " << tippoint[1] << ", " << tippoint[2] << std::endl;
            }

            PointCloudInputPtr pointsonplane1_processed(new PointCloudInput);
            PointCloudInputPtr pointsonplane2_processed(new PointCloudInput);

            Eigen::Affine3f rot = Eigen::Affine3f::Identity();
            // TODO need to consider terminalcylindercoeffs_, maybe need to prepare another parameter to define the pose to cable
            rot.matrix().block<3,1>(0,1) = -dir;

            if (inlierindices_upper1->indices.size() < indicessize_protruding) {
                if (inlierindices_upper2->indices.size() < indicessize_protruding) {
                    // plane1でもplane2でもでっぱりなし
                    // compute area size
                    // TODO? should evaluate totalarea1 itself?
                    /*{{{*/
                    proj.setModelType (pcl::SACMODEL_PLANE);
                    proj.setInputCloud (terminalscenepoints);
                    proj.setIndices (inlierindices_enlarged1);
                    proj.setModelCoefficients (planemodelcoeffs1);
                    proj.filter (*pointsonplane1_processed);
                    chull.setInputCloud (pointsonplane1_processed);
                    chull.setDimension(2);
                    chull.setComputeAreaVolume(true);
                    chull.reconstruct (*pointsonplane1_processed);
                    double totalarea1 = chull.getTotalArea();
                    proj.setModelType (pcl::SACMODEL_PLANE);
                    proj.setIndices (inlierindices_enlarged2);
                    proj.setInputCloud (terminalscenepoints2);
                    proj.setModelCoefficients (planemodelcoeffs2);
                    proj.filter (*pointsonplane2_processed);
                    chull.setInputCloud (pointsonplane2_processed);
                    chull.setDimension(2);
                    chull.setComputeAreaVolume(true);
                    chull.reconstruct (*pointsonplane2_processed);
                    double totalarea2 = chull.getTotalArea();/*}}}*/
                    std::cout << "totalarea1: " << totalarea1 << " totalarea2: " << totalarea2 << std::endl;
                    if (totalarea1 > totalarea2) {
                        // plane1の方が大きい
                        rot.matrix().block<3,1>(0,2) = -n1;
                        rot.matrix().block<3,1>(0,0) = rot.matrix().block<3,1>(0,1).cross(rot.matrix().block<3,1>(0,2));//n2;
                        terminaltransform = Eigen::Translation<float, 3>(tippoint[0], tippoint[1], tippoint[2]) * rot;
                        if(n2.dot(rot.matrix().block<3,1>(0,0)) > 0) {
                            pcl::console::print_highlight("terminal pose estimation pattern1-1\n");
                            terminaltransform *= Eigen::Translation<float, 3>(-x_offset, y_offset, z_offset);
                        } else {
                            pcl::console::print_highlight("terminal pose estimation pattern1-2\n");
                            terminaltransform *= Eigen::Translation<float, 3>(x_offset, y_offset, z_offset);
                        }
                    } else {
                        // plane2の方が大きい
                        rot.matrix().block<3,1>(0,2) = -n2;
                        rot.matrix().block<3,1>(0,0) = rot.matrix().block<3,1>(0,1).cross(rot.matrix().block<3,1>(0,2));
                        terminaltransform = Eigen::Translation<float, 3>(tippoint[0], tippoint[1], tippoint[2]) * rot;
                        if(n1.dot(rot.matrix().block<3,1>(0,0)) > 0) {
                            pcl::console::print_highlight("terminal pose estimation pattern2-1\n");
                            terminaltransform *= Eigen::Translation<float, 3>(-x_offset, y_offset, z_offset);
                        } else {
                            pcl::console::print_highlight("terminal pose estimation pattern2-2\n");
                            terminaltransform *= Eigen::Translation<float, 3>(x_offset, y_offset, z_offset);
                        }
                    }
                } else {
                    // plane2ででっぱりあり
                    Eigen::Vector3f newn2 = dir.cross(n1);
                    if(newn2.dot(n2) < 0) {
                        newn2[0] *= -1; newn2[1] *= -1; newn2[2] *= -1;
                    }
                    rot.matrix().block<3,1>(0,2) = newn2;
                    rot.matrix().block<3,1>(0,0) = rot.matrix().block<3,1>(0,1).cross(rot.matrix().block<3,1>(0,2));
                    terminaltransform = Eigen::Translation<float, 3>(tippoint[0], tippoint[1], tippoint[2]) * rot;
                    if(n1.dot(rot.matrix().block<3,1>(0,0)) > 0) {
                        pcl::console::print_highlight("terminal pose estimation pattern3-1\n");
                        terminaltransform *= Eigen::Translation<float, 3>(-x_offset, y_offset, -z_offset);
                    } else {
                        pcl::console::print_highlight("terminal pose estimation pattern3-2\n");
                        terminaltransform *= Eigen::Translation<float, 3>(x_offset, y_offset, -z_offset);
                    }
                }
            } else {
                if (inlierindices_upper2->indices.size() < indicessize_protruding) {
                    // plane1ででっぱりあり、plane2ででっぱりなし
                    rot.matrix().block<3,1>(0,2) = n1;
                    rot.matrix().block<3,1>(0,0) = rot.matrix().block<3,1>(0,1).cross(rot.matrix().block<3,1>(0,2));//n2;
                    terminaltransform = Eigen::Translation<float, 3>(tippoint[0], tippoint[1], tippoint[2]) * rot;
                    if(n2.dot(rot.matrix().block<3,1>(0,0)) > 0) {
                        pcl::console::print_highlight("terminal pose estimation pattern4-1\n");
                        terminaltransform *= Eigen::Translation<float, 3>(-x_offset, y_offset, -z_offset);
                    } else {
                        pcl::console::print_highlight("terminal pose estimation pattern4-2\n");
                        terminaltransform *= Eigen::Translation<float, 3>(x_offset, y_offset, -z_offset);
                    }
                } else {
                    // plane1ででもplane2ででっぱりあり おかしい
                    pcl::console::print_highlight("both plane1 and plane2 are protruding? weird\n");
                    pcl::console::print_highlight("ignore plane2");
                    goto estimatefromplane1;
                }
            }
        } else {
estimatefromplane1:
            viewer_->removeShape("plane2"+terminalindex);

            PointCloudInputPtr pointsonplane1_projected(new PointCloudInput);
            PointCloudInputPtr pointsonplane1_projected_onlyhead(new PointCloudInput);
            PointCloudInputPtr pointsonplane1_chull(new PointCloudInput);
            pcl::console::print_highlight("estimate from plane1\n");
            //pcl::io::savePCDFileBinaryCompressed ("pointsonplane1.pcd", *pointsonplane1);
            proj.setModelType (pcl::SACMODEL_PLANE);
            //proj.setIndices (inlierindices_enlarged1);
            proj.setIndices (inlierindices1);
            proj.setInputCloud (terminalscenepoints);
            proj.setModelCoefficients (planemodelcoeffs1);
            proj.filter (*pointsonplane1_projected);
            //pcl::io::savePCDFileBinaryCompressed ("pointsonplane1_projected.pcd", *pointsonplane1_projected);

            //chull.setInputCloud (pointsonplane1_projected);
            //chull.setDimension(2);
            //chull.setComputeAreaVolume(true);
            //chull.reconstruct (*pointsonplane1_chull);
            //double totalarea1 = chull.getTotalArea();

            // guess tip point
            Eigen::Vector3f origdir, origpt;
            origdir = Eigen::Vector3f(terminalcoeffs->values[3], terminalcoeffs->values[4], terminalcoeffs->values[5]);
            origpt = Eigen::Vector3f(terminalcoeffs->values[0], terminalcoeffs->values[1], terminalcoeffs->values[2]);
            // use pca to get better dir,pt
            pcl::PCA<PointNT> pca;
            pca.setInputCloud(terminalscenepoints);
            Eigen::Vector3f pcapt3;
            Eigen::Vector4f pcapt4;
            pcl::compute3DCentroid (*terminalscenepoints, pcapt4);
            pcapt3 = pcapt4.segment<3>(0);
            Eigen::Vector3f pcadir = pca.getEigenVectors().col(0);
            if (pcadir.dot(origdir) < 0) {
                origdir = -pcadir;
            } else {
                origdir = pcadir;
            }
            origpt = pcapt3;
            origdir.normalize(); // just in case
            float maxtiplength, mintiplength;
            Eigen::Vector3f maxtippoint, maxtippointonline, mintippoint, mintippointonline;
            //std::vector<int> removedpointindices;
            //pcl::removeNaNFromPointCloud<PointNT>(*pointsonplane1_projected, removedpointindices);
            computeTipPointByProjectIntoLine<PointNT>(pointsonplane1_projected, origdir, origpt, maxtiplength, maxtippoint, maxtippointonline, mintiplength, mintippoint, mintippointonline);
            PointCloudInputPtr tmpcloud(new PointCloudInput);
            tmpcloud->width = 4;
            tmpcloud->height = 1;
            tmpcloud->points.resize(4);
            tmpcloud->points[0].x = maxtippoint[0]; tmpcloud->points[0].y = maxtippoint[1]; tmpcloud->points[0].z = maxtippoint[2];
            tmpcloud->points[1].x = maxtippointonline[0]; tmpcloud->points[1].y = maxtippointonline[1]; tmpcloud->points[1].z = maxtippointonline[2];
            tmpcloud->points[2].x = mintippoint[0]; tmpcloud->points[2].y = mintippoint[1]; tmpcloud->points[2].z = mintippoint[2];
            tmpcloud->points[3].x = mintippointonline[0]; tmpcloud->points[3].y = mintippointonline[1]; tmpcloud->points[3].z = mintippointonline[2];
            //viewer_->removePointCloud("tippoints"+terminalindex);
            //viewer_->addPointCloud<PointNT>(tmpcloud, pcl::visualization::PointCloudColorHandlerCustom<PointNT> (tmpcloud, 30, 200, 100),  "tippoints"+terminalindex);
            //viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 30, "tippoints"+terminalindex);

            Eigen::Vector3f origpt_shifted = origpt + (maxtiplength - terminalcoeffs->values[7]) * origdir;

            // slide terminalcoeffs to get only head pointcloud
            std::cout << "slide terminalcoeffs to get only head pointcloud" <<std::endl;
            pcl::ModelCoefficients::Ptr terminalcoeffs_onlyhead(new pcl::ModelCoefficients);
            std::copy(terminalcoeffs->values.begin(), terminalcoeffs->values.end(), std::back_inserter(terminalcoeffs_onlyhead->values));
            terminalcoeffs_onlyhead->values[0] = origpt_shifted[0];
            terminalcoeffs_onlyhead->values[1] = origpt_shifted[1];
            terminalcoeffs_onlyhead->values[2] = origpt_shifted[2];
            terminalcoeffs_onlyhead->values[8] = terminalcoeffs->values[7] - flatheadoffset;
            std::cout << "terminalcoeffs_onlyhead: " <<std::endl;
            std::cout << *terminalcoeffs_onlyhead << std::endl;

            // extract head pointcloud
            pcl::PointIndices::Ptr terminalheadindices(new pcl::PointIndices);
            size_t numheadpoints = FindPointIndicesInsideCylinder<PointNT>(pointsonplane1_projected, terminalcoeffs_onlyhead, terminalheadindices);
            if (numheadpoints == 0) {
                PCL_ERROR("no points inside the head of the terminal!?\n");
                return false;
            }
            extract.setInputCloud (pointsonplane1_projected);
            extract.setIndices (terminalheadindices);
            extract.setNegative (false);
            extract.filter (*pointsonplane1_projected_onlyhead);
            //pcl::RadiusOutlierRemoval<PointNT> rorfilter;
            //rorfilter.setRadiusSearch(radiusoutlierremovalsize);
            //rorfilter.setInputCloud(pointsonplane1_projected_onlyhead);
            //rorfilter.filter(*pointsonplane1_projected_onlyhead);

            ///  compute rectangle vertices ///
            // recompute origpt using only head pointcloud
            origdir = Eigen::Vector3f(terminalcoeffs->values[3], terminalcoeffs->values[4], terminalcoeffs->values[5]);
            origpt = Eigen::Vector3f(terminalcoeffs->values[0], terminalcoeffs->values[1], terminalcoeffs->values[2]);
            origdir.normalize();
            //float maxtiplength, mintiplength;
            //Eigen::Vector3f maxtippoint, maxtippointonline, mintippoint, mintippointonline;
            computeTipPointByProjectIntoLine<PointNT>(pointsonplane1_projected_onlyhead, origdir, origpt, maxtiplength, maxtippoint, maxtippointonline, mintiplength, mintippoint, mintippointonline);
            origpt_shifted = origpt + (maxtiplength - terminalcoeffs->values[7]) * origdir;
            
            // compute new dir and new pt using minAreaRect
            std::cout << "compute new dir and new pt using minAreaRect" << std::endl;
            Eigen::Vector3f newdir, newpt;
            std::vector<Eigen::Vector3f> rectvertices = minAreaRect<PointNT>(planemodelcoeffs1, pointsonplane1_projected_onlyhead);// the last one is the center
            // (pickup 2 head vertices of the rectvertices)
            std::vector<std::pair<double, unsigned int> > pairlengthtopointindex;
            for (size_t irectpoint = 0; irectpoint < 4; irectpoint++) {
                Eigen::Vector3f& pt = rectvertices[irectpoint];
                pairlengthtopointindex.push_back(std::pair<double, unsigned int>(((pt.dot (origdir) - origpt_shifted.dot (origdir)) / origdir.dot (origdir)), irectpoint));
            }
            std::sort(pairlengthtopointindex.begin(), pairlengthtopointindex.end(), std::greater<std::pair<double, unsigned int> >());
            Eigen::Vector3f headpoints0 = rectvertices[pairlengthtopointindex[0].second];
            Eigen::Vector3f headpoints1 = rectvertices[pairlengthtopointindex[1].second];
            Eigen::Vector3f headpoints2 = rectvertices[pairlengthtopointindex[2].second];
            Eigen::Vector3f headpoints3 = rectvertices[pairlengthtopointindex[3].second];
            Eigen::Vector3f rectanglecenter = rectvertices[4];

            // recompute rectvertices --------
            size_t pointsinsidecyl = 0;
            pcl::ModelCoefficients::Ptr rectverticesline0model(new pcl::ModelCoefficients);
            pcl::ModelCoefficients::Ptr rectverticesline1model(new pcl::ModelCoefficients);
            PointCloudInputPtr rectverticesline0points(new PointCloudInput);
            PointCloudInputPtr rectverticesline1points(new PointCloudInput);
            Eigen::Vector3f rectverticesline0dir = (headpoints1 - headpoints0).normalized();//head
            Eigen::Vector3f rectverticesline1dir = (headpoints3 - headpoints2).normalized();//tail
            rectverticesline0model->values.resize(9);
            rectverticesline0model->values[0] = headpoints0[0];
            rectverticesline0model->values[1] = headpoints0[1];
            rectverticesline0model->values[2] = headpoints0[2];
            rectverticesline0model->values[3] = rectverticesline0dir[0];
            rectverticesline0model->values[4] = rectverticesline0dir[1];
            rectverticesline0model->values[5] = rectverticesline0dir[2];
            rectverticesline0model->values[6] = 0.002;
            rectverticesline0model->values[7] = (headpoints1 - headpoints0).norm();
            rectverticesline0model->values[8] = 0;
            rectverticesline1model->values.resize(9);
            rectverticesline1model->values[0] = headpoints2[0];
            rectverticesline1model->values[1] = headpoints2[1];
            rectverticesline1model->values[2] = headpoints2[2];
            rectverticesline1model->values[3] = rectverticesline1dir[0];
            rectverticesline1model->values[4] = rectverticesline1dir[1];
            rectverticesline1model->values[5] = rectverticesline1dir[2];
            rectverticesline1model->values[6] = 0.002;
            rectverticesline1model->values[7] = (headpoints3 - headpoints2).norm();
            rectverticesline1model->values[8] = 0;
            pcl::PointIndices::Ptr rectverticesline0indices(new pcl::PointIndices);
            pcl::PointIndices::Ptr rectverticesline1indices(new pcl::PointIndices);
            float maxtiplength0, mintiplength0, maxtiplength1, mintiplength1;
            Eigen::Vector3f maxtippoint0, maxtippointonline0, mintippoint0, mintippointonline0;
            Eigen::Vector3f maxtippoint1, maxtippointonline1, mintippoint1, mintippointonline1;
            pointsinsidecyl = FindPointIndicesInsideCylinder<PointNT>(pointsonplane1_projected_onlyhead, rectverticesline0model, rectverticesline0indices);
            do {
            if (pointsinsidecyl == 0) {
                std::cout << "skiprecomputerectangle0" << std::endl;
                break;
                //goto skiprecomputerectangle;
            }
            pointsinsidecyl = FindPointIndicesInsideCylinder<PointNT>(pointsonplane1_projected_onlyhead, rectverticesline1model, rectverticesline1indices);
            if (pointsinsidecyl == 0) {
                std::cout << "skiprecomputerectangle1" << std::endl;
                break;
                //goto skiprecomputerectangle;
            }
            extract.setInputCloud (pointsonplane1_projected_onlyhead);
            extract.setIndices (rectverticesline0indices);
            extract.setNegative (false);
            extract.filter (*rectverticesline0points);
            computeTipPointByProjectIntoLine<PointNT>(rectverticesline0points, rectverticesline0dir, headpoints0, maxtiplength0, maxtippoint0, maxtippointonline0, mintiplength0, mintippoint0, mintippointonline0);

            extract.setInputCloud (pointsonplane1_projected_onlyhead);
            extract.setIndices (rectverticesline1indices);
            extract.setNegative (false);
            extract.filter (*rectverticesline1points);
            computeTipPointByProjectIntoLine<PointNT>(rectverticesline1points, rectverticesline1dir, headpoints2, maxtiplength1, maxtippoint1, maxtippointonline1, mintiplength1, mintippoint1, mintippointonline1);

            float headtotaillengthratio = (maxtippoint0 - mintippoint0).norm() / (maxtippoint1 - mintippoint1).norm();
            std::cout << "headtotaillengthratio: " << headtotaillengthratio << std::endl;
            if ( headtotaillengthratio < 0.9 || 1.1 < headtotaillengthratio) {
                std::cout << "the estimated head not a rectangle anymore. skiprecomputerectangle1." << std::endl;
                break;
            }

            headpoints0 = maxtippoint0;
            headpoints1 = mintippoint0;
            headpoints2 = maxtippoint1;
            headpoints3 = mintippoint1;

            // visualization
            pcl::PointCloud<pcl::PointXYZ>::Ptr rectcloudnew(new pcl::PointCloud<pcl::PointXYZ>);
            rectcloudnew->width = 5;
            rectcloudnew->height = 5;
            rectcloudnew->points.resize(rectcloudnew->width*rectcloudnew->height);
            rectcloudnew->points[0].x = maxtippoint0(0); rectcloudnew->points[0].y = maxtippoint0(1); rectcloudnew->points[0].z = maxtippoint0(2);
            rectcloudnew->points[1].x = mintippoint0(0); rectcloudnew->points[1].y = mintippoint0(1); rectcloudnew->points[1].z = mintippoint0(2);
            rectcloudnew->points[2].x = maxtippoint1(0); rectcloudnew->points[2].y = maxtippoint1(1); rectcloudnew->points[2].z = maxtippoint1(2);
            rectcloudnew->points[3].x = mintippoint1(0); rectcloudnew->points[3].y = mintippoint1(1); rectcloudnew->points[3].z = mintippoint1(2);
            rectanglecenter = (maxtippoint0 + mintippoint0 + maxtippoint1 + mintippoint1)/4.0;
            rectcloudnew->points[4].x = rectanglecenter(0);
            rectcloudnew->points[4].y = rectanglecenter(1);
            rectcloudnew->points[4].z = rectanglecenter(2);

            //pcl::PointXYZ pt0new, pt1new;
            //pt0new.x = newpt[0]; pt0new.y = newpt[1]; pt0new.z = newpt[2];
            //pt1new.x = newpt[0] + newdir[0] * 0.01; pt1new.y = newpt[1] + newdir[1] * 0.01; pt1new.z = newpt[2] + newdir[2] * 0.01;
            if (enablefancyvisualization) {
                viewer_->removePointCloud("rectverticesnew"+terminalindex);
                viewer_->addPointCloud<pcl::PointXYZ>(rectcloudnew, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> (rectcloudnew, 0, 90, 50),  "rectverticesnew"+terminalindex);
                viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 25, "rectverticesnew"+terminalindex);
                //viewer_->removeShape("termianlcylnew" + terminalindex);
                //viewer_->addLine(pt0new, pt1new, 0, 1.0, 0.5, "termianlcylnew" + terminalindex);
            }

            // end of recompute rectvertices --------
            } while (false);

//skiprecomputerectangle:
            newdir = (headpoints0 + headpoints1)/2.0 - (headpoints2 + headpoints3)/2.0;
            newdir.normalize();
            newpt = (headpoints0 + headpoints1)/2.0 - y_offset * newdir;
            std::cout << "newdir: " << newdir << std::endl;
            std::cout << "newpt: " << newpt << std::endl;


            // visualization
            pcl::PointCloud<pcl::PointXYZ>::Ptr rectcloud(new pcl::PointCloud<pcl::PointXYZ>);
            rectcloud->width = 5;
            rectcloud->height = 5;
            rectcloud->points.resize(rectcloud->width*rectcloud->height);
            for (size_t irv = 0; irv < rectvertices.size() ; irv++) {
                pcl::PointXYZ& pt = rectcloud->points[irv];
                pt.x = rectvertices[irv][0];
                pt.y = rectvertices[irv][1];
                pt.z = rectvertices[irv][2];
            }
            pcl::PointXYZ pt0, pt1;
            pt0.x = newpt[0]; pt0.y = newpt[1]; pt0.z = newpt[2];
            pt1.x = newpt[0] + newdir[0] * 0.01; pt1.y = newpt[1] + newdir[1] * 0.01; pt1.z = newpt[2] + newdir[2] * 0.01;

            if (enablefancyvisualization) {
                viewer_->removePointCloud("rectvertices"+terminalindex);
                viewer_->addPointCloud<pcl::PointXYZ>(rectcloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> (rectcloud, 100, 90, 30),  "rectvertices"+terminalindex);
                viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 25, "rectvertices"+terminalindex);
                viewer_->removeShape("termianlcyl" + terminalindex);
                viewer_->addLine(pt0, pt1, 1, 0, 0.5, "termianlcyl" + terminalindex);
            }
            // end of visualization

            // compute area using rectvertices
            double headarea = (rectvertices[0] - rectvertices[1]).norm() * (rectvertices[1] - rectvertices[2]).norm();

            Eigen::Affine3f rot = Eigen::Affine3f::Identity();
            rot.matrix().block<3,1>(0,1) = -newdir;
            std::cout << "head plane size is " << headarea << ", ";
            std::cout << " > " << areasize_threshold << "?" << std::endl;
            if (headarea > areasize_threshold) {
                if (inlierindices_upper1->indices.size() < indicessize_protruding) {
                    // でっぱりが奥
                    pcl::console::print_highlight("terminal pose estimation pattern5-1\n");
                    rot.matrix().block<3,1>(0,2) = -Eigen::Vector3f(planemodelcoeffs1->values[0], planemodelcoeffs1->values[1], planemodelcoeffs1->values[2]);
                    rot.matrix().block<3,1>(0,0) = rot.matrix().block<3,1>(0,1).cross(rot.matrix().block<3,1>(0,2));
                    terminaltransform = Eigen::Translation<float, 3>(newpt[0], newpt[1], newpt[2]) * rot * Eigen::Translation<float, 3>(0, 0, z_offset);
                } else {
                    // でっぱりが手前
                    pcl::console::print_highlight("terminal pose estimation pattern5-2\n");
                    rot.matrix().block<3,1>(0,2) = Eigen::Vector3f(planemodelcoeffs1->values[0], planemodelcoeffs1->values[1], planemodelcoeffs1->values[2]);
                    rot.matrix().block<3,1>(0,0) = rot.matrix().block<3,1>(0,1).cross(rot.matrix().block<3,1>(0,2));
                    terminaltransform = Eigen::Translation<float, 3>(newpt[0], newpt[1], newpt[2]) * rot * Eigen::Translation<float, 3>(0, 0, -z_offset);
                }
            } else {
                // distinguish which side is upper side
                Eigen::Vector3f planenormal(planemodelcoeffs1->values[0], planemodelcoeffs1->values[1], planemodelcoeffs1->values[2]);
                Eigen::Affine3f tmptransform = Eigen::Affine3f::Identity();
                tmptransform.matrix().block<3,1>(0,0) = planenormal;
                tmptransform.matrix().block<3,1>(0,1) = -newdir;
                tmptransform.matrix().block<3,1>(0,2) = tmptransform.matrix().block<3,1>(0,0).cross(tmptransform.matrix().block<3,1>(0,1));
                tmptransform.matrix().block<3,1>(0,3) = newpt;
                std::cout << "tmptransform: " << std::endl;
                std::cout << tmptransform.matrix() << std::endl;
                PointCloudInputPtr pointsonplane1_origin(new PointCloudInput);
                pcl::transformPointCloud(*pointsonplane1_projected, *pointsonplane1_origin, tmptransform.inverse());

                if (enablefancyvisualization) {
                    viewer_->removePointCloud("pointsonplane1_origin"+terminalindex);
                    viewer_->addPointCloud(pointsonplane1_origin, pcl::visualization::PointCloudColorHandlerCustom<PointNT> (pointsonplane1_origin, 30, 130, 80),  "pointsonplane1_origin"+terminalindex);
                    viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 20, "pointsonplane1_origin"+terminalindex);
                }

                // count num of points above/below x-y plane
                size_t abovepoints = 0, belowpoints = 0;
                for (size_t ipoint = 0; ipoint < pointsonplane1_origin->points.size(); ipoint++) {
                    if (pointsonplane1_origin->points[ipoint].y > flatheadoffset) {
                        if (pointsonplane1_origin->points[ipoint].z > z_offset){
                            abovepoints++;
                        } else if (pointsonplane1_origin->points[ipoint].z < -z_offset) {
                            belowpoints++;
                        }
                    }
                }
                std::cout << "above points: " << abovepoints << " below points: " << belowpoints << std::endl;
                terminaltransform = tmptransform;
                if (abovepoints > belowpoints) {
                    // でっぱりが上
                    pcl::console::print_highlight("terminal pose estimation pattern6-1\n");
                    terminaltransform *= Eigen::Translation<float, 3>(-x_offset, 0, 0);
                } else {
                    // でっぱりが下
                    pcl::console::print_highlight("terminal pose estimation pattern6-2\n");
                    terminaltransform.matrix().block<3,1>(0,0) *= -1;
                    terminaltransform.matrix().block<3,1>(0,2) *= -1;
                    terminaltransform *= Eigen::Translation<float, 3>(x_offset, 0, 0);
                }

            }
        }
        std::cout << terminaltransform.matrix() << std::endl;
        viewer_->addCoordinateSystem(0.05, terminaltransform);
        detectedterminaltransform = terminaltransform;
        return true;
    }


    // lock viewer occasionally
    void findCables(std::vector<Cable>& cables)/*{{{*/
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
            viewer_->spinOnce();
        }

        std::vector<std::pair<int, int> > scannedpointindicescache; // cableindex, sliceindex
        scannedpointindicescache.resize(input_->points.size());
        std::fill (scannedpointindicescache.begin(),scannedpointindicescache.end(),std::pair<int, int>(-1, -1));
        std::vector<int> ignoreindices;
        ignoreindices.resize(1);

        int cableindex;
        for (size_t isample = 0; isample < sampled_indices.points.size(); isample++) {
            cableindex = cables.size();
            bool alreadyscanned = (scannedpointindicescache[sampled_indices.points[isample]].first != -1);
            std::stringstream textss;
            textss << "sampled_pt_" << isample;
            if (!!viewer_) {
                boost::mutex::scoped_lock lock(viewer_mutex_);
                viewer_->removeText3D(textss.str());
                viewer_->addText3D (textss.str(), sampledpoints->points[isample], 0.001);
                viewer_->spinOnce();
            }
            if (not alreadyscanned) {
                pcl::PointXYZ eachpt;
                eachpt.x = sampledpoints->points[isample].x;
                eachpt.y = sampledpoints->points[isample].y;
                eachpt.z = sampledpoints->points[isample].z;

                pcl::console::print_highlight(std::string(std::string("-----  find cable from ") + textss.str() + std::string("-----\n")).c_str());
                ignoreindices[0] = cableindex;
                std::vector<EndPointInfo> endpointinfos(2);
                Cable cable = findCableFromPoint(eachpt, cables, endpointinfos, scannedpointindicescache, ignoreindices);

                if (cable.size() == 0) {
                    std::cout << "empty cable. skip"  << std::endl;
                    continue;
                }
                // comment out to disable cable reconnection
/**/
                int startintersectionsize[2] = {0,0};
                int endintersectionsize[2] = {0,0};
                bool swapcable[2] = {false, false};
                int leastintersectionsize = 5;
                if (endpointinfos[0].cablesliceindex != -1) {
                    std::cout << "!! may be class " << endpointinfos[0].cablesliceindex << "?" << std::endl;
                    std::cout << "!! endpointinfos[0]: " << endpointinfos[0].pointindices.size() << std::endl;
                    Cable existingcable = *boost::next(cables.begin(), endpointinfos[0].cablesliceindex); // an existing cable which locates at the start point of the detected cable
                    int cablelength = existingcable.size();
                    for (size_t ipoint = 0; ipoint < endpointinfos[0].pointindices.size(); ipoint++) {
                        if (scannedpointindicescache[endpointinfos[0].pointindices[ipoint]].second == 0)
                        {
                            startintersectionsize[0]++;
                        }
                        if (scannedpointindicescache[endpointinfos[0].pointindices[ipoint]].second == cablelength-1)
                        {
                            endintersectionsize[0]++;
                        }
                    }
                    std::cout << "!! " << endpointinfos[0].cablesliceindex << ", " << cablelength << ", " << startintersectionsize[0] << ", " << endintersectionsize[0] << std::endl;
                    if (startintersectionsize[0] > leastintersectionsize)
                    {
                        // [(end) ... existing (start)] <-> [(start) current ... (end)]
                        Eigen::Vector3f axis1((*cable.begin())->cylindercoeffs->values[3],(*cable.begin())->cylindercoeffs->values[4],(*cable.begin())->cylindercoeffs->values[5]);
                        Eigen::Vector3f axis2((*existingcable.begin())->cylindercoeffs->values[3],(*existingcable.begin())->cylindercoeffs->values[4],(*existingcable.begin())->cylindercoeffs->values[5]);
                        int flipdir = -1;
                        if (axis1.dot(axis2) > 0 ) {
                            flipdir = 1; // for the case when the length of the existing cable is 1
                        }
                        for (typename std::list<CableSlicePtr>::iterator itrslice = existingcable.begin(); itrslice != existingcable.end(); ++itrslice) {
                            (*itrslice)->cylindercoeffs->values[3]*= flipdir;
                            (*itrslice)->cylindercoeffs->values[4]*= flipdir;
                            (*itrslice)->cylindercoeffs->values[5]*= flipdir;
                            cable.push_front(*itrslice);
                        }
                        swapcable[0] = true;
                    }
                    else if (endintersectionsize[0] > leastintersectionsize)
                    {
                        // [(start) ... existing (end)] <-> [(start) current ... (end)]
                        for (typename std::list<CableSlicePtr>::reverse_iterator itrslice = existingcable.rbegin(); itrslice != existingcable.rend(); ++itrslice) {
                            cable.push_front(*itrslice);
                        }
                        swapcable[0] = true;
                    }
                }

                if (endpointinfos[1].cablesliceindex != -1) {
                    std::cout << "!! may be class " << endpointinfos[1].cablesliceindex << "?" << std::endl;
                    std::cout << "!! endpointinfos[1]: " << endpointinfos[1].pointindices.size() << std::endl;
                    Cable existingcable = *boost::next(cables.begin(), endpointinfos[1].cablesliceindex); // an existing cable which locates at the start point of the detected cable
                    int cablelength = existingcable.size();
                    for (size_t ipoint = 0; ipoint < endpointinfos[1].pointindices.size(); ipoint++) {
                        if (scannedpointindicescache[endpointinfos[1].pointindices[ipoint]].second == 0)
                        {
                            startintersectionsize[1]++;
                        }
                        if (scannedpointindicescache[endpointinfos[1].pointindices[ipoint]].second == cablelength-1)
                        {
                            endintersectionsize[1]++;
                        }
                    }
                    std::cout << "!! " << endpointinfos[1].cablesliceindex << ", " << cablelength << ", " << startintersectionsize[1] << ", " << endintersectionsize[1] << std::endl;
                    if (startintersectionsize[1] > leastintersectionsize)
                    {
                        // [(start) ... current (end)] <-> [(start) existing ... (end)]
                        Eigen::Vector3f axis1((*(boost::prior(cable.end())))->cylindercoeffs->values[3],(*(boost::prior(cable.end())))->cylindercoeffs->values[4],(*(boost::prior(cable.end())))->cylindercoeffs->values[5]);
                        Eigen::Vector3f axis2((*existingcable.begin())->cylindercoeffs->values[3],(*existingcable.begin())->cylindercoeffs->values[4],(*existingcable.begin())->cylindercoeffs->values[5]);
                        int flipdir = 1;
                        if (axis1.dot(axis2) < 0 ) {
                            flipdir = -1;
                        }
                        for (typename std::list<CableSlicePtr>::iterator itrslice = existingcable.begin(); itrslice != existingcable.end(); ++itrslice) {
                            cable.push_back(*itrslice);
                        }
                        swapcable[1] = true;
                    }
                    else if (endintersectionsize[1] > leastintersectionsize)
                    {
                        // [(start) ... current (end)] <-> [(end) existing ... (start)]
                        for (typename std::list<CableSlicePtr>::reverse_iterator itrslice = existingcable.rbegin(); itrslice != existingcable.rend(); ++itrslice) {
                            (*itrslice)->cylindercoeffs->values[3]*= -1;
                            (*itrslice)->cylindercoeffs->values[4]*= -1;
                            (*itrslice)->cylindercoeffs->values[5]*= -1;
                            cable.push_back(*itrslice);
                        }
                        swapcable[1] = true;
                    }
                }

                // update cache
                int currentcableindex;
                if (!swapcable[0] && !swapcable[1]) {
                    currentcableindex = cableindex;
                }
                else if (swapcable[0] && !swapcable[1]) {
                    currentcableindex = endpointinfos[0].cablesliceindex;
                }
                else if (!swapcable[0] && swapcable[1]) {
                    currentcableindex = endpointinfos[1].cablesliceindex;
                }
                else if (swapcable[0] && swapcable[1]) {
                    currentcableindex = std::min(endpointinfos[0].cablesliceindex, endpointinfos[1].cablesliceindex);
                }

                std::cout << cables.size() << std::endl;
                std::cout << (*std::max_element(scannedpointindicescache.begin(), scannedpointindicescache.end(), pairCompare1<int>)).first << std::endl;
                //
                int islice = 0;
                for (typename std::list<CableSlicePtr>::iterator sliceitr = cable.begin(); sliceitr!= cable.end(); ++sliceitr, ++islice) {
                    for (std::vector<int>::const_iterator pointindexitr = (*sliceitr)->searchedindices->indices.begin(); pointindexitr != (*sliceitr)->searchedindices->indices.end(); pointindexitr++ ) {
                        
                        scannedpointindicescache[(*pointindexitr)] = std::pair<int, int>(currentcableindex, islice);
                    }
                }
                std::cout << (*std::max_element(scannedpointindicescache.begin(), scannedpointindicescache.end(), pairCompare1<int>)).first << std::endl;
                //
                if (!swapcable[0] && !swapcable[1])
                {
                    //if (cable.size() > 1) {
                    //if (1) {  // NOTE: for debug
                    cables.resize(cables.size()+1);
                    (*(cables.end()-1)).swap(cable);
                    std::cout << "<<< append cable " << cableindex << std::endl;
                }
                else if (swapcable[0] && !swapcable[1]) {
                    (*boost::next(cables.begin(), endpointinfos[0].cablesliceindex)).swap(cable);
                    std::cout << "<<< swap cable at " << endpointinfos[0].cablesliceindex<< std::endl;
                }
                else if (!swapcable[0] && swapcable[1]) {
                    (*boost::next(cables.begin(), endpointinfos[1].cablesliceindex)).swap(cable);
                    std::cout << "<<< swap cable at " << endpointinfos[1].cablesliceindex<< std::endl;
                }
                else if (swapcable[0] && swapcable[1]) {
                    int swapedcableindex = std::min(endpointinfos[0].cablesliceindex, endpointinfos[1].cablesliceindex);
                    int erasedcableindex = std::max(endpointinfos[0].cablesliceindex, endpointinfos[1].cablesliceindex);
                    (*boost::next(cables.begin(), swapedcableindex)).swap(cable);
                    typename std::vector<Cable>::iterator itrcable = cables.begin();
                    std::advance(itrcable, erasedcableindex);
                    cables.erase(itrcable);
                    int islice = 0;
                    for (typename std::vector<std::pair<int, int> >::iterator itrpicache = scannedpointindicescache.begin(); itrpicache!= scannedpointindicescache.end(); ++itrpicache) {
                        if (itrpicache->first > erasedcableindex) {
                            itrpicache->first -= 1;
                        }
                    }
                    std::cout << "<<< swap cable at " << swapedcableindex << " and erase " << erasedcableindex << std::endl;
                }
/**/
/*
                int islice = 0;
                for (typename std::list<CableSlicePtr>::iterator sliceitr = cable.begin(); sliceitr!= cable.end(); ++sliceitr, ++islice) {
                    for (std::vector<int>::const_iterator pointindexitr = (*sliceitr)->searchedindices->indices.begin(); pointindexitr != (*sliceitr)->searchedindices->indices.end(); pointindexitr++ ) {
                        
                        scannedpointindicescache[(*pointindexitr)] = std::pair<int, int>(cableindex, islice);
                    }
                }
                //if (1) {  // NOTE: for debug
                if (cable.size() >= 1) {
                    cables.resize(cables.size()+1);
                    (*(cables.end()-1)).swap(cable);
                    std::cout << "<<< append cable " << cableindex << std::endl;
                }
*/
            }
        }
    }/*}}}*/

    class EndPointInfo
    {
    public:
        EndPointInfo() : cablesliceindex(-1) {
            cablesliceindex = -1;
        }
        int cablesliceindex;
        std::vector<int> pointindices;
    };

    // note: viewer lock free
    Cable findCableFromPoint(pcl::PointXYZ point, std::vector<Cable>& cables, std::vector<EndPointInfo>& endpointinfos, const std::vector<std::pair<int, int> >& scannedpointindicescache = std::vector<std::pair<int, int> >(), const std::vector<int>& ignoreindices = std::vector<int>()) { /*{{{*/
        pcl::PointIndices::Ptr k_indices;
        Cable cable;
        CableSlicePtr slice, oldslice, baseslice;
        slice.reset(new CableSlice());
        oldslice.reset(new CableSlice());

        k_indices = _findClosePointsIndices(point);
        pcl::PointXYZ pt;
        bool cableslicefound = _estimateCylinderAroundPointsIndices(k_indices, *slice);
        if (!cableslicefound) {
            PCL_INFO("[first search] no valid slice found, break.\n");
            //NOTE: for debug
            //std::copy(k_indices->indices.begin(), k_indices->indices.end(), std::back_inserter(slice->searchedindices->indices));
            //cable.push_front(slice);
            return cable;
        }
        oldslice = slice; baseslice = slice;
        std::cout << "first slice found!" << std::endl;
        cable.push_back(slice);

        endpointinfos.resize(2);
        std::vector<int> scannedindices;
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

                k_indices = _findClosePointsIndices(pt);
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

            if (scannedpointindicescache.size() != 0)
            {
                int scannedindex = _isAlreadyScanned(k_indices, scannedpointindicescache, scannedindices, ignoreindices);
                if( scannedindex != -1)
                {
                    PCL_INFO("[forward search] one of close points is already scanned, break.\n");
                    endpointinfos[0].cablesliceindex = scannedindex;
                    endpointinfos[0].pointindices.swap(scannedindices);
                    break;
                }
            }
            slice.reset(new CableSlice());
            slice->centerpt_= pt;
            Eigen::Vector3f initialaxis(oldslice->cylindercoeffs->values[3], oldslice->cylindercoeffs->values[4], oldslice->cylindercoeffs->values[5]);
            cableslicefound = _estimateCylinderAroundPointsIndices(k_indices, *slice, pt, initialaxis, 1);
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

                k_indices = _findClosePointsIndices(pt);
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

            if (scannedpointindicescache.size() != 0)
            {
                int scannedindex = _isAlreadyScanned(k_indices, scannedpointindicescache, scannedindices, ignoreindices);
                if( scannedindex != -1)
                {
                    PCL_INFO("[backward search] one of close points is already scanned, break.\n");
                    endpointinfos[1].cablesliceindex = scannedindex;
                    endpointinfos[1].pointindices.swap(scannedindices);
                    break;
                }
            }
            slice.reset(new CableSlice());
            slice->centerpt_ = pt;
            Eigen::Vector3f initialaxis(oldslice->cylindercoeffs->values[3], oldslice->cylindercoeffs->values[4], oldslice->cylindercoeffs->values[5]);
            cableslicefound = _estimateCylinderAroundPointsIndices(k_indices, *slice, pt, initialaxis, 1);
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

    /*
     * return cable index if one of the points indicated by indices is already scanned in some cable
     */    
    int _isAlreadyScanned(pcl::PointIndices::Ptr indices, const std::vector<std::pair<int, int> >& scannedpointindicescache, std::vector<int>& scannedindices, const std::vector<int>& ignoreindices = std::vector<int>())/*{{{*/
    {
        /*{{{*/
        /*
        for (typename std::vector<Cable>::const_iterator itr = cables.begin(); itr != cables.end(); itr++) {
            for (typename std::list<CableSlicePtr>::const_iterator sliceitr = (*itr).begin(); sliceitr!= (*itr).end(); ++sliceitr) {
                for (std::vector<int>::iterator i = indices->indices.begin(); i != indices->indices.end(); i++) {
                    std::vector<int>::iterator founditr 
                        = std::find( (*sliceitr)->searchedindices->indices.begin(), (*sliceitr)->searchedindices->indices.end() , (*i) );
                    if ( founditr !=(*sliceitr)->searchedindices->indices.end() ) {
                        return true;
                    }
                }
            }
        }
        return false;
        *//*}}}*/
        bool ignoreit = false;
        std::map<int, int> histgram;
        std::map<int, std::vector<int> > mapcableindextopointsindices;
        for (std::vector<int>::iterator i = indices->indices.begin(); i != indices->indices.end(); i++) {
            if (scannedpointindicescache[*i].first != -1) {
                ignoreit = false;
                for (std::vector<int>::const_iterator j = ignoreindices.begin(); j != ignoreindices.end(); j++) {
                    if ((*j) == scannedpointindicescache[*i].first) {
                        ignoreit = true;
                        break;
                    }
                }
                if (!ignoreit) {
                    //if(histgram.find(scannedpointindicescache[*i].first) == histgram.end()) { histgram[scannedpointindicescache[*i].first] = 0; }
                    histgram[scannedpointindicescache[*i].first] += 1;
                    mapcableindextopointsindices[scannedpointindicescache[*i].first].push_back(*i);
                }
            }
        }

        if (histgram.size() == 0) {
            return -1;
        }
        std::map<int, int>::iterator maxitr = std::max_element(histgram.begin(), histgram.end(), pairCompare2<int>);
        scannedindices.swap(mapcableindextopointsindices[maxitr->first]);
        return maxitr->first;
    }/*}}}*/

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

            /* uncomment this line if you want to visualize the points which are detected as cable */
            /*
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
            */

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

    void findCableTerminal(Cable& cable, double offset) {/*{{{*/
        if (cable.size() < 3) {
            pcl::console::print_highlight("no enough cable slices to find terminals. break.");
            return;
        }
        //CableSlicePtr firstslice = cable.front();
        //CableSlicePtr lastslice  = cable.back();
        //
        pcl::ModelCoefficients::Ptr firstterminalcoeffs(new pcl::ModelCoefficients(*terminalcylindercoeffs_));
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
        firstterminalcoeffs->values[7] += 0.010;
        firstterminalcoeffs->values[8] -= 0.005;

        Eigen::Affine3f terminaltransform;
        _estimateTerminalFromInitialCoeffs(firstterminalcoeffs, terminaltransform);
    }/*}}}*/

    bool _estimateTerminalFromInitialCoeffs(pcl::ModelCoefficients::Ptr terminalcoeffs, Eigen::Affine3f& terminaltransform, std::string terminalindex = "")/*{{{*/
    {
        pcl::PointIndices::Ptr terminalscenepointsindices(new pcl::PointIndices());
        /*
        pcl::ModelCoefficients::Ptr cylmodel(new pcl::ModelCoefficients());
        cylmodel->values.resize(7);
        for (int i = 0; i < 7; i++) {
            cylmodel->values[i] = terminalcoeffs->values[i];
        }
        viewer_->removeShape("cylinder");
        viewer_->addCylinder(*cylmodel);
        */
        Eigen::Affine3f besttransform;
        size_t points = _findScenePointIndicesInsideCylinder(terminalcoeffs, terminalscenepointsindices);
        if (points < 30) {
            PCL_INFO("too few points at the  slice to detect terminal, points: %d\n", points);
        } else {
            pcl::ExtractIndices<PointNT> extract;
            PointCloudInputPtr terminalscenepoints(new PointCloudInput());
            extract.setInputCloud (input_);
            extract.setIndices (terminalscenepointsindices);
            //extract.setNegative (true);
            extract.filter (*terminalscenepoints);

            // visualize points/*{{{*/
            //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgbfield(extractedpoints);
            pcl::visualization::PointCloudColorHandlerCustom<PointNT> rgbfield(terminalscenepoints, 255, 0, 255);

            viewer_->removePointCloud ("extracted_points" + terminalindex);
            viewer_->addPointCloud<PointNT> (terminalscenepoints, rgbfield,  "extracted_points" + terminalindex);
            viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "extracted_points" + terminalindex);
            std::cout << terminalcoeffs->values[0] << " "
                << terminalcoeffs->values[1] << " "
                << terminalcoeffs->values[2] << " " << std::endl;
            std::cout << terminalcoeffs->values[3] << " "
                << terminalcoeffs->values[4] << " "
                << terminalcoeffs->values[5] << " " << std::endl;
            std::cout << terminalcoeffs->values[6] << " "
                << terminalcoeffs->values[7] << " "
                << terminalcoeffs->values[8] << " " << std::endl;
            //viewer_->addCylinder(*cylmodel);
            // end of visualize points/*}}}*/

            pcl::console::print_highlight ("Downsampling...\n");
            pcl::VoxelGrid<PointNT> grid;
            PointCloudInputPtr voxelterminalcloud(new PointCloudInput());
            double leafsize = 0.0005;
            grid.setLeafSize (leafsize, leafsize, leafsize);
            grid.setInputCloud (terminalscenepoints);
            grid.filter (*terminalscenepoints);
            //grid.setInputCloud (terminalcloud_);
            //grid.filter (*voxelterminalcloud);

            // compute cross product
            Eigen::Vector3f v(terminalcylindercoeffs_->values[3],terminalcylindercoeffs_->values[4],terminalcylindercoeffs_->values[5]);
            Eigen::Vector3f dir(terminalcoeffs->values[3],terminalcoeffs->values[4],terminalcoeffs->values[5]);
            Eigen::Vector3f rotaxis = v.cross(dir);
            double rottheta = asin(rotaxis.norm());
            rotaxis.normalize();
            //Eigen::Transform<float, 3, Eigen::Affine> t; //same as the following one
            Eigen::Affine3f t;
            t = Eigen::Translation<float, 3>(terminalcoeffs->values[0], terminalcoeffs->values[1], terminalcoeffs->values[2]) * Eigen::AngleAxisf(rottheta, rotaxis);
            //viewer_->addCoordinateSystem(0.1, t);

            PointCloudInputPtr transformedterminalcloud(new PointCloudInput);
            PointCloudInputPtr transformedterminalcloud_best(new PointCloudInput);

            //pcl::copyPointCloud(*object, *object_xyz);
            typename pcl::KdTreeFLANN<PointNT>::Ptr kdtree (new pcl::KdTreeFLANN<PointNT>());
            pcl::ExtractIndices<PointNT> extractNT;
            int maxinliernum = 0;
            double maxscore = 0.0;
            int rotnum = 360/5;
            Eigen::Affine3f rotaffine;
            rotaffine = Eigen::AngleAxisf(2.0*M_PI/rotnum,v);
            for (size_t irot = 0; irot < rotnum; irot++) {
                Eigen::Affine3f tnew;
                tnew = t;
                for (size_t i = 0; i < irot; i++) {
                    tnew = tnew * rotaffine;
                }
                pcl::transformPointCloud (*terminalcloud_, *transformedterminalcloud, tnew);
                PointCloudInputPtr transformedterminalcloud_voxel(new PointCloudInput);
                // cut out hidden points
                {
                    pcl::VoxelGridOcclusionEstimation<PointNT> vgoe;
                    vgoe.setInputCloud(transformedterminalcloud);
                    vgoe.setLeafSize(leafsize, leafsize, leafsize);
                    vgoe.initializeVoxelGrid();
                    *transformedterminalcloud_voxel = vgoe.getFilteredPointCloud();
                    pcl::PointIndices::Ptr visibleindices(new pcl::PointIndices);
                    for (size_t ipoint = 0; ipoint < transformedterminalcloud_voxel->points.size(); ipoint++) {
                        PointNT& pt = transformedterminalcloud_voxel->points[ipoint];
                        Eigen::Vector3i ijk = vgoe.getGridCoordinates(pt.x, pt.y, pt.z);
                        int out_state;
                        int status = vgoe.occlusionEstimation(out_state, ijk); //0 = free, 1 = occluded
                        if (out_state == 0) {
                            visibleindices->indices.push_back(ipoint);
                        }
                    }
                    extractNT.setInputCloud (transformedterminalcloud_voxel);
                    extractNT.setIndices (visibleindices);
                    //extractNT.setNegative (true);
                    extractNT.filter (*transformedterminalcloud);
                }

                // compute inliers
                kdtree->setInputCloud(transformedterminalcloud);
                std::vector<int> k_indices;
                std::vector<float> k_sqr_distances;
                PointNT pt;
                std::vector<int> inliers;
                for (size_t iscenepoint = 0; iscenepoint < terminalscenepoints->points.size(); iscenepoint++) {
                    pt.x = terminalscenepoints->points[iscenepoint].x;
                    pt.y = terminalscenepoints->points[iscenepoint].y;
                    pt.z = terminalscenepoints->points[iscenepoint].z;
                    kdtree->radiusSearch (pt, leafsize*0.5, k_indices, k_sqr_distances);
                    if (k_indices.size() > 0) {
                        inliers.push_back(iscenepoint);
                    }
                }
                viewer_->removePointCloud("object_transformed" + terminalindex);
                viewer_->addPointCloud(transformedterminalcloud, ColorHandlerNT (transformedterminalcloud, 255.0, 0.0, 0.0), "object_transformed" + terminalindex);
                viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "object_transformed" + terminalindex);
                viewer_->spinOnce();
                boost::this_thread::sleep(boost::posix_time::milliseconds(100));
                pcl::console::print_highlight("irot: %d, inliers: %d/%d(scene), %d(visible object), %d(object)\n", irot, inliers.size(), terminalscenepoints->points.size(), transformedterminalcloud->points.size(), transformedterminalcloud_voxel->points.size());
                if (inliers.size() > maxinliernum) {
                    besttransform = tnew;
                    maxinliernum = inliers.size();
                    pcl::copyPointCloud(*transformedterminalcloud, *transformedterminalcloud_best);
                }
            }
            viewer_->addCoordinateSystem(0.05, besttransform);
            pcl::transformPointCloudWithNormals(*terminalcloud_, *transformedterminalcloud, besttransform);
            viewer_->removePointCloud("object_transformed" + terminalindex);
            viewer_->addPointCloud(transformedterminalcloud, ColorHandlerNT (transformedterminalcloud, 255.0, 0.0, 0.0), "object_transformed" + terminalindex);
            //////////////////////////////////
            // icp
/*{{{*/
/*
            //pcl::IterativeClosestPointWithNormals<PointNT, PointNT> icp;
            pcl::GeneralizedIterativeClosestPoint<PointNT, PointNT> icp;
            size_t rotnum = 4;
            Eigen::Affine3f rotaffine;
            rotaffine = Eigen::AngleAxisf(2.0*M_PI/rotnum,v);

            //icp.setInputTarget(transformedterminalcloud);
            // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
            //icp.setMaxCorrespondenceDistance (0.02);
            // Set the maximum number of iterations (criterion 1)
            icp.setMaximumIterations (100);
            // Set the transformation epsilon (criterion 2)
            icp.setTransformationEpsilon (0.00001); //(1e-8);
            //icp.setRotationEpsilon (0.001);
            // Set the euclidean distance difference epsilon (criterion 3)
            //icp.setEuclideanFitnessEpsilon (0.000001); //(1);

            icp.setRANSACOutlierRejectionThreshold(0.0005);


            for (size_t irot = 0; irot < rotnum; irot++) {
                pcl::console::print_highlight ("Starting terminal alignment... (%d)\n", irot);
                Eigen::Affine3f tnew;
                tnew = t;
                for (size_t i = 0; i < irot; i++) {
                    tnew = tnew * rotaffine;
                }

                viewer_->removeCoordinateSystem();
                viewer_->addCoordinateSystem(0.03, tnew);
                pcl::copyPointCloud<PointNT>(*voxelterminalcloud, *transformedterminalcloud);
                pcl::transformPointCloudWithNormals (*voxelterminalcloud, *transformedterminalcloud, tnew);
                pcl::io::savePCDFileBinaryCompressed ("terminalscenepoints.pcd", *terminalscenepoints);
                std::stringstream filess;
                filess << "transformedterminalcloud" << irot << ".pcd";
                pcl::io::savePCDFileBinaryCompressed (filess.str(), *transformedterminalcloud);

                // cut out hidden points
                {
                    PointCloudInputPtr flattenedtransformedterminalcloud(new PointCloudInput);
                    pcl::copyPointCloud(*transformedterminalcloud, *flattenedtransformedterminalcloud);
                    for (size_t i = 0; i < flattenedtransformedterminalcloud->points.size(); i++) {
                        flattenedtransformedterminalcloud->points[i].z = 0.0;
                    }

                    pcl::PointIndices::Ptr visibleindices(new pcl::PointIndices());
                    std::vector<bool> visiblecandidateindices;
                    visiblecandidateindices.resize(transformedterminalcloud->points.size());
                    std::fill(visiblecandidateindices.begin(), visiblecandidateindices.end(), true);

                    typename pcl::KdTreeFLANN<PointNT>::Ptr kdtree (new pcl::KdTreeFLANN<PointNT>());
                    kdtree->setInputCloud(flattenedtransformedterminalcloud);
                    std::vector<int> k_indices;
                    std::vector<float> k_sqr_distances;

                    float zvisiblethreshold = 6*leafsize;
                    for (size_t i = 0; i < flattenedtransformedterminalcloud->points.size(); i++) {

                        if(visiblecandidateindices[i]) {
                            kdtree->radiusSearch (flattenedtransformedterminalcloud->points[i], leafsize, k_indices, k_sqr_distances);
                            for (size_t j = 0; j < k_indices.size(); j++) {
                                if (transformedterminalcloud->points[i].z - transformedterminalcloud->points[j].z > zvisiblethreshold) {
                                    visiblecandidateindices[i] = false;
                                }
                                if (transformedterminalcloud->points[i].z - transformedterminalcloud->points[j].z < zvisiblethreshold) {
                                    visiblecandidateindices[j] = false;
                                }
                            }
                        }
                    }

                    for (size_t i = 0; i < visiblecandidateindices.size(); i++) {
                        if (visiblecandidateindices[i]) {
                            visibleindices->indices.push_back(i);
                        }
                    }
                    extract.setInputCloud (transformedterminalcloud);
                    extract.setIndices (visibleindices);
                    //extract.setNegative (true);
                    extract.filter (*transformedterminalcloud);
                    std::stringstream filess2;
                    filess2 << "visible_transformedterminalcloud" << irot << ".pcd";
                    pcl::io::savePCDFileBinaryCompressed (filess2.str(), *transformedterminalcloud);
                }

                //icp.setInputCloud(terminalscenepoints);
                //icp.setInputTarget(transformedterminalcloud);
                icp.setInputCloud(transformedterminalcloud);
                icp.setInputTarget(terminalscenepoints);
                PointCloudInput finalpoints;
                icp.align(finalpoints);//, tnew.matrix());
                std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;

                if (icp.hasConverged ())
                {
                    // Print results
                    printf ("\n");
                    Eigen::Matrix4f transformation = icp.getFinalTransformation () ;
                    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
                    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
                    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
                    pcl::console::print_info ("\n");
                    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
                    pcl::console::print_info ("\n");
                    //pcl::console::print_info ("Inliers: %i/%i\n", icp.getInliers ().size (), voxelterminalcloud->size ());

                    // Show alignment
                    //std::cout << tnew.matrix() << std::endl;
                    PointCloudInputPtr terminalcloudforvis(new PointCloudInput());
                    terminalcloudforvis->points.resize(terminalcloud_->points.size());
                    //pcl::visualization::PointCloudColorHandlerRGBField<PointNT> rgbfield(terminalcloudforvis);
                    pcl::visualization::PointCloudColorHandlerCustom<PointNT> rgbfield(terminalcloudforvis, 0, 255, 0);
                    pcl::transformPointCloudWithNormals (*voxelterminalcloud, *terminalcloudforvis, tnew * transformation);
                    viewer_->removePointCloud("terminal");
                    viewer_->addPointCloud<PointNT> (terminalcloudforvis, rgbfield, "terminal");
                    viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "terminal");

                    //vtkSmartPointer<vtkMatrix4x4> vtkmatrix = vtkSmartPointer<vtkMatrix4x4>::New();
                    //vtkSmartPointer<vtkTransform> vtktransformation = vtkSmartPointer<vtkTransform>::New();
                    //pcl::visualization::PCLVisualizer::convertToVtkMatrix(tnew*transformation, vtkmatrix);
                    //vtktransformation->SetMatrix(&(*vtkmatrix));
                    //viewer_->removeShape("PLYModel");
                    //viewer_->addModelFromPLYFile (std::string("/home/sifi/Dropbox/mujin/cad/lancable_terminal/lancable_terminal_fine_m.ply"), vtktransformation);

                }
                else
                {
                    pcl::console::print_error ("Alignment failed!\n");
                    //vtkSmartPointer<vtkMatrix4x4> vtkmatrix = vtkSmartPointer<vtkMatrix4x4>::New();
                    //vtkSmartPointer<vtkTransform> vtktransformation = vtkSmartPointer<vtkTransform>::New();
                    //for (size_t row = 0; row < 4; row++) {
                    //    for (size_t col = 0; col < 4; col++) {
                    //        vtkmatrix->SetElement(row,col,t(row,col));
                    //    }
                    //}
                    ////pcl::visualization::PCLVisualizer::convertToVtkMatrix(t, vtkmatrix);
                    //vtktransformation->SetMatrix(&(*vtkmatrix));
                    //viewer_->removeShape("PLYModel");
                    //viewer_->addModelFromPLYFile (std::string("/home/sifi/Dropbox/mujin/lancable_terminal/lancable_terminal_fine_m.ply"), vtktransformation);

                    std::cout << tnew.matrix() << std::endl;
                    pcl::visualization::PointCloudColorHandlerCustom<PointNT> rgbfield(transformedterminalcloud, 0, 255, 0);
                    //viewer_->addPointCloud<PointNT> (terminalcloudforvis, rgbfield, "terminal");
                    //viewer_->addPointCloud<PointNT> (object_aligned, rgbfield, "terminal");
                    viewer_->removePointCloud("failedterminal");
                    viewer_->addPointCloud<PointNT> (transformedterminalcloud, rgbfield, "failedterminal");
                    viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "failedterminal");

                    std::stringstream filess;
                    filess << "transformedterminalcloud" << irot << ".pcd";
                    pcl::io::savePCDFileBinaryCompressed (filess.str(), *transformedterminalcloud);
                    viewer_->spinOnce();
                    //boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
                    viewer_->spinOnce();
                    boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
                }
            }
            */
/*}}}*/
            // alignment
            ///////////////////////////////////////
/*{{{*/
/*
            FeatureCloudT::Ptr terminal_features (new FeatureCloudT); //should be a class member
            FeatureCloudT::Ptr scene_features (new FeatureCloudT); //should be a class member
            FeatureEstimationT fest;
            
            fest.setRadiusSearch (0.002);
            fest.setInputCloud (terminalscenepoints);
            fest.setInputNormals (terminalscenepoints);
            fest.compute (*scene_features);
            fest.setInputCloud (voxelterminalcloud);
            fest.setInputNormals (voxelterminalcloud);
            fest.compute (*terminal_features);
            
            pcl::console::print_highlight ("Starting terminal alignment...\n");
            pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
            PointCloudInputPtr object_aligned(new PointCloudInput());

            size_t rotnum = 4;
            align.setInputSource (voxelterminalcloud);
            align.setSourceFeatures (terminal_features);
            align.setInputTarget (terminalscenepoints);
            align.setTargetFeatures (scene_features);
            align.setMaximumIterations (10000); // Number of RANSAC iterations
            align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
            align.setCorrespondenceRandomness (2); // Number of nearest features to use
            align.setSimilarityThreshold (0.9f); // Polygonal edge length similarity threshold
            align.setMaxCorrespondenceDistance(1.5f * leafsize); // Inlier threshold
            align.setInlierFraction (0.1f);//(0.25f); // Required inlier fraction for accepting a pose hypothesis

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
                viewer_->addCoordinateSystem(0.03, tnew);
                //pcl::copyPointCloud<PointNT>(*voxelterminalcloud, *transformedterminalcloud);
                //pcl::transformPointCloudWithNormals (*voxelterminalcloud, *transformedterminalcloud, tnew);

                {
                    pcl::ScopeTime t("Alignment");
                    align.align (*object_aligned, tnew.matrix());
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
                    PointCloudInputPtr terminalcloudforvis(new PointCloudInput());
                    pcl::visualization::PointCloudColorHandlerRGBField<PointNT> rgbfield(terminalcloud_);
                    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBNormal> rgbfield(extractedpoints, 0, 255, 0);
                    pcl::copyPointCloud(terminalcloud_, terminalcloudforvis);

                    pcl::transformPointCloudWithNormals (*terminalcloud_, *terminalcloudforvis, transformation);
                    pcl::visualization::PointCloudColorHandlerCustom<PointNT> rgbfield(object_aligned, 0, 255, 0);
                    viewer_->addPointCloud<PointNT> (terminalcloudforvis, rgbfield, "terminal");
                    //viewer_->addPointCloud<PointNT> (object_aligned, rgbfield, "terminal");
                    viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "terminal");

                    //vtkSmartPointer<vtkMatrix4x4> vtkmatrix = vtkSmartPointer<vtkMatrix4x4>::New();
                    //vtkSmartPointer<vtkTransform> vtktransformation = vtkSmartPointer<vtkTransform>::New();
                    //pcl::visualization::PCLVisualizer::convertToVtkMatrix(transformation, vtkmatrix);
                    //vtktransformation->SetMatrix(&(*vtkmatrix));
                    //std::stringstream plyname;
                    //plyname << "plymodel_" << rand();
                    //viewer_->addModelFromPLYFile (std::string("/home/sifi/Dropbox/mujin/cad/lancable_terminal/lancable_terminal_fine_m.ply"),
                            //vtktransformation, plyname.str());
                    break;
                }
                else
                {
                    pcl::console::print_error ("Alignment failed!\n");
                    //vtkSmartPointer<vtkMatrix4x4> vtkmatrix = vtkSmartPointer<vtkMatrix4x4>::New();
                    //vtkSmartPointer<vtkTransform> vtktransformation = vtkSmartPointer<vtkTransform>::New();
                    //for (size_t row = 0; row < 4; row++) {
                    //    for (size_t col = 0; col < 4; col++) {
                    //        vtkmatrix->SetElement(row,col,tnew(row,col));
                    //    }
                    //}
                    //vtktransformation->SetMatrix(&(*vtkmatrix));
                    //viewer_->removeShape("PLYModel");
                    //viewer_->addModelFromPLYFile (std::string("/home/sifi/Dropbox/mujin/cad/lancable_terminal/lancable_terminal_fine_m.ply"), vtktransformation);
                    std::cout << tnew.matrix() << std::endl;
                    pcl::visualization::PointCloudColorHandlerCustom<PointNT> rgbfield(transformedterminalcloud, 0, 255, 0);
                    //viewer_->addPointCloud<PointNT> (terminalcloudforvis, rgbfield, "terminal");
                    //viewer_->addPointCloud<PointNT> (object_aligned, rgbfield, "terminal");
                    viewer_->removePointCloud("failedterminal");
                    pcl::transformPointCloudWithNormals(*voxelterminalcloud, *transformedterminalcloud, tnew);
                    viewer_->addPointCloud<PointNT> (transformedterminalcloud, "failedterminal");
                    viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "failedterminal");

                    std::stringstream filess;
                    filess << "transformedterminalcloud" << irot << ".pcd";
                    pcl::io::savePCDFileBinaryCompressed (filess.str(), *transformedterminalcloud);
                    viewer_->spinOnce();
                    //boost::this_thread::sleep(boost::posix_time::milliseconds(1000));

                }
            }
*/
/*}}}*/
        }

        terminaltransform = besttransform;
    }/*}}}*/

    /* 
     * param[in] terminalcylindercoeffs: terminalcylindercoeffs
     * param[out] indices: scene point indices inside terminal cylinder
     * return: the size of indices
     */
    size_t _findScenePointIndicesInsideCylinder(pcl::ModelCoefficients::Ptr terminalcylindercoeffs, pcl::PointIndices::Ptr indices)/*{{{*/
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

    pcl::PointIndices::Ptr _findClosePointsIndices(pcl::PointXYZ pt, double radius = 0) { /*{{{*/
        if (radius == 0) {
            radius = cableradius_*2;
        }

        pcl::PointIndices::Ptr k_indices(new pcl::PointIndices());
        std::vector<float> k_sqr_distances;

        if (!(pcl_isfinite (pt.x) && pcl_isfinite (pt.y) && pcl_isfinite (pt.z))) {
            return k_indices;
        }
        //kdtree_->setInputCloud(points_);
        kdtree_->radiusSearch (pt, radius, k_indices->indices, k_sqr_distances);
        return k_indices;
    } /*}}}*/

    bool _estimateCylinderAroundPointsIndices (pcl::PointIndices::Ptr pointsindices, CableSlice& slice, pcl::PointXYZ centerpt = pcl::PointXYZ(), const Eigen::Vector3f& initialaxis = Eigen::Vector3f(), double eps_angle=0.0) /*{{{*/
    {
        pcl::ScopeTime t("_estimateCylinderAroundPointsIndices");
        // Create the segmentation object
        pcl::SACSegmentationFromNormals<PointNT, PointNT> seg;
        pcl::PointIndices::Ptr cylinderinlierindices(new pcl::PointIndices());
        // Optional
        seg.setOptimizeCoefficients (true); // TODO this may be unnecessary
        Eigen::Vector3f axis;
        axis = initialaxis;
        axis.normalize();
        //TODO 'setRadiusLimits' instead of setAxis and setEpsAngle
        seg.setAxis (axis); 
        seg.setEpsAngle(eps_angle);
        // Mandatory
        seg.setModelType (pcl::SACMODEL_CYLINDER);
        //seg.setMethodType (pcl::SAC_RANSAC);
        seg.setMethodType (pcl::SAC_RRANSAC);
        seg.setMaxIterations(5000);
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
        std::stringstream ss;
        ss << "    cylinder Model cylindercoeffs: "
                  << slice.cylindercoeffs->values[0] << " "
                  << slice.cylindercoeffs->values[1] << " "
                  << slice.cylindercoeffs->values[2] << " "
                  << slice.cylindercoeffs->values[3] << " "
                  << slice.cylindercoeffs->values[4] << " "
                  << slice.cylindercoeffs->values[5] << " "
                  << slice.cylindercoeffs->values[6] << " ";
        PCL_INFO(std::string(ss.str() + "\n").c_str());
        return _validateCableSlice(slice);
    } /*}}}*/
    bool _validateCableSlice (CableSlice& slice) /*{{{*/
    {
        PCL_INFO("[_validateCableSlice] radius: %f, given: %f\n", slice.cylindercoeffs->values[6], cableradius_);
        //if(cableradius_* 0.65 > slice.cylindercoeffs->values[6] || slice.cylindercoeffs->values[6] > cableradius_* 1.35 ) {
        if(cableradius_* 0.6 > slice.cylindercoeffs->values[6] || slice.cylindercoeffs->values[6] > cableradius_* 1.4 ) {
        //if(cableradius_* 0.3 > slice.cylindercoeffs->values[6] || slice.cylindercoeffs->values[6] > cableradius_* 1.7 ) {
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
     * 7,8    : upperheight, lowerheight(negative value)
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

    pcl::PointCloud<pcl::PointWithRange>::Ptr cloudcenters_;
};


} // namespace pcl_cable_detection
#endif /* end of include guard */

