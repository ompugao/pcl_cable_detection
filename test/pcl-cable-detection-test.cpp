#include <iostream>
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/utility.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/lexical_cast.hpp>
#include "pcl-cable-detection.h"
#include <pcl/io/ply_io.h>
#include <sstream>
#include <zmq.hpp>
#include <vector>
#include <opencv2/opencv.hpp>
#include <picojson.h>
#include <pcl/range_image/range_image_planar.h>

class ImageProcessingResult
{
public:
    int label;
    double centroid[2];
    int minx;
    int maxx;
    int miny;
    int maxy;
};

class ZmqClient/*{{{*/
{
public:
    ZmqClient(std::string uri)
    {
        _uri = uri;
        _InitializeSocket();
    }

    virtual ~ZmqClient()
    {
        _DestroySocket();
    }

    unsigned int Recv(std::vector<ImageProcessingResult>& results) {
        std::string rep;
        std::stringstream repss;
        unsigned int size = this->Recv(rep);
        std::cerr << rep << std::endl;
        repss << rep;
        picojson::value v;
        repss >> v;
        std::string err = picojson::get_last_error();
        if (! err.empty()) {
            std::cerr << err << std::endl;
            return -1;
        }
        results.resize(0);
        picojson::object pcrep = v.get<picojson::object>();
        picojson::array pcres = pcrep["result"].get<picojson::array>();
        for (picojson::array::iterator itr = pcres.begin(); itr != pcres.end(); ++itr) {
            ImageProcessingResult res;
            picojson::array a = itr->get<picojson::array>();
            res.label = (int)(a[0].get<double>());
            res.centroid[0] = a[1].get<double>();
            res.centroid[1] = a[2].get<double>();
            res.minx = (int)(a[3].get<double>());
            res.maxx = (int)(a[4].get<double>());
            res.miny = (int)(a[5].get<double>());
            res.maxy = (int)(a[6].get<double>());
            results.push_back(res);
        }
        return size;
    }

    unsigned int Recv(std::string& data)
    {
        zmq::message_t reply;
        _socket->recv(&reply);
        //std::string replystring((char *) reply.data(), (size_t) reply.size());
        data.resize(reply.size());
        std::copy((uint8_t*)reply.data(), (uint8_t*)reply.data()+reply.size(), data.begin());
        return reply.size(); //replystring;
    }

    void Send(const cv::Mat& image)
    {
        picojson::object req;

        req["image"] = picojson::value(picojson::array());
        req["image"].get<picojson::array>().resize(image.rows);
        for(int i = 0; i < image.rows; i++) {
            req["image"].get<picojson::array>()[i] = picojson::value(picojson::array());
            req["image"].get<picojson::array>()[i].get<picojson::array>().resize(image.cols);
        }
        for(int i = 0; i < image.rows; i++ ) {
            for(int j = 0; j < image.cols; j++) {
                req["image"].get<picojson::array>()[i].get<picojson::array>()[j] = picojson::value((double)image.at<uchar>(i,j));
            }
        }
        this->Send(picojson::value(req).serialize());
    }

    void Send(const std::string& message)
    {
        zmq::message_t request(message.size());
        memcpy((void *) request.data(), message.c_str(), message.size());
        _socket->send(request);
    }


protected:
    void _InitializeSocket()
    {
        _context.reset(new zmq::context_t (1));
        _socket.reset(new zmq::socket_t ((*(zmq::context_t*)_context.get()), ZMQ_REQ));
        _socket->connect (_uri.c_str());
    }

    void _DestroySocket()
    {
        if (!!_socket) {
            _socket->close();
            _socket.reset();
        }
        if( !!_context ) {
            _context->close();
            _context.reset();
        }
    }

    std::string _uri;
    boost::shared_ptr<zmq::context_t> _context;
    boost::shared_ptr<zmq::socket_t>  _socket;
};/*}}}*/

void LoadDavidCameraParameter(std::string cameracalfilename, int& width, int& height, boost::property_tree::ptree& tsaiparam_pt, Eigen::Affine3f& Tcamtoworld)/*{{{*/
{
    using namespace boost::property_tree;
    ptree pt;
    std::cout << "reading file..." << std::endl;
    read_xml(cameracalfilename, pt);

    if (boost::optional<std::string> camera_model = pt.get_optional<std::string>("camera_model")) {
        std::cout << camera_model.get() << std::endl;
    }
    std::string tsaiparameterstrings[] = {"cx","cy", "f","sx", "kappa1"};
    for (size_t i = 0; i < 5; ++i) {
        if (boost::optional<double> num = pt.get_optional<double>(tsaiparameterstrings[i])) {
            std::cout << tsaiparameterstrings[i] << ": " << num.get() << std::endl;
            tsaiparam_pt.put<double>(tsaiparameterstrings[i], num.get());
        }
    }
    if (boost::optional<int> num = pt.get_optional<int>("resX")) {
        width = num.get();
        std::cout << "width: " << width << std::endl;
    }
    if (boost::optional<int> num = pt.get_optional<int>("resY")) {
        height = num.get();
        std::cout << "height: " << height << std::endl;
    }


    std::string colstr[] = {"n","o","a","p"};
    std::string rowstr[] = {"x","y","z","w"};
    for (size_t icol = 0; icol < 4; icol++) {
        for (size_t irow = 0; irow < 4; irow++) {
            if (boost::optional<double> num = pt.get_optional<double>("Pose." + colstr[icol] + rowstr[irow])) {
                Tcamtoworld(irow,icol) = num.get();
            } else {
                std::cerr << "Pose." + colstr[icol] + rowstr[irow] + " not found" << std::endl;
            }
        }
    }
    Tcamtoworld(0,3) /= 1000.0;
    Tcamtoworld(1,3) /= 1000.0;
    Tcamtoworld(2,3) /= 1000.0;
    std::cout << "!!!!matrix!!!!" << std::endl;
    std::cout << Tcamtoworld.matrix() << std::endl;
}/*}}}*/

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

using namespace pcl_cable_detection;

int main (int argc, char** argv)
{
    // command line arguments parsing /*{{{*/
    namespace bpo = boost::program_options;
    namespace bfs = boost::filesystem;
    bpo::options_description opts_desc("Allowed options");
    bpo::positional_options_description p;

    opts_desc.add_options()
        ("input_file", bpo::value< std::string >(), "Input pcd.")
        ("input_imagefile", bpo::value< std::string >(), "Input image.")
        ("cameracalfile", bpo::value< std::string >(), "david camera cal file (full path).")
        ("removeplane", bpo::value<bool>()->default_value(false), "findplane or not?")
        ("voxelsize_findplane", bpo::value< double >(), "voxelsize_findplane")
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
    std::string input_imagefile = opts["input_imagefile"].as<std::string> ();
    std::string cameracalfile = opts["cameracalfile"].as<std::string> ();

    double voxelsize_findplane = opts["voxelsize_findplane"].as<double>();
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
    std::cout << "loading image file..." << std::endl;
    cv::Mat image;
    image = cv::imread(input_imagefile,0); //read as gray-scale image
/*}}}*/
    if(removeplane) { /*{{{*/
        std::cout << "finding plane..." << std::endl;
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        if (!pcl_cable_detection::findPlane<pcl::PointNormal>(cloud, voxelsize_findplane, distthreshold_findplane, *inliers, *coefficients)) {
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
    //std::cout << "compute curvature..." << std::endl; /*{{{*/
    /*
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
    */
/*}}}*/

    // image processing
    std::cout << "image processing..." << std::endl;
    //ZmqClient client("ipc:///tmp/imageprocessing.ipc");
    ZmqClient client("tcp://127.0.0.1:59010");
    client.Send(image);
    std::vector<ImageProcessingResult> results;
    client.Recv(results);

    // point cloud processing
    pcl::console::setVerbosityLevel (pcl::console::L_INFO);
    pcl::PointCloud<pcl::PointNormal>::Ptr terminalcloud(new pcl::PointCloud<pcl::PointNormal>());
    pcl::io::loadPLYFile(cableterminalply, *terminalcloud);
    terminalcloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
    Eigen::Vector3f axis(0,-1,0);
    typedef CableDetection<pcl::PointNormal> CableDetectionPointNormal;
    CableDetectionPointNormal cabledetection(terminalcloud, axis);
    cabledetection.SetUpViewer();

    pcl::console::setVerbosityLevel (pcl::console::L_DEBUG);
    cabledetection.setCableRadius(cableradius);
    cabledetection.setThresholdCylinderModel(distthreshold_cylindermodel);
    cabledetection.setSceneSamplingRadius(scenesamplingradius);
    //cabledetection.setInputCloud(cloud_normals);
    cabledetection.setInputCloud(cloud_filtered);
    int width; int height; boost::property_tree::ptree tsaiparam_pt; Eigen::Affine3f Tcamtoworld;
    LoadDavidCameraParameter(cameracalfile, width, height, tsaiparam_pt, Tcamtoworld);
    pcl::RangeImagePlanar rip;
    rip.createFromPointCloudWithFixedSize(*(cabledetection.input_), width, height, tsaiparam_pt.get<double>("cx"), tsaiparam_pt.get<double>("cy"),
            tsaiparam_pt.get<double>("f"), tsaiparam_pt.get<double>("f"), Eigen::Affine3f::Identity());
    pcl::PointCloud<pcl::PointWithRange>::Ptr cloudcenters(new pcl::PointCloud<pcl::PointWithRange>);
    for (std::vector<ImageProcessingResult>::iterator itrres = results.begin(); itrres != results.end(); ++itrres) {
        pcl::PointWithRange ptrange = rip.getPoint((float)itrres->centroid[0], (float)itrres->centroid[1]);
        std::cout << ptrange << std::endl;
        ptrange.z += 0.0042;
        cloudcenters->points.push_back(ptrange);
    }
    cloudcenters->width = results.size();
    cloudcenters->height = results.size();
    cloudcenters->is_dense = false;

    typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> ColorHandlerTR;
    cabledetection.viewer_->addPointCloud(cloudcenters, ColorHandlerTR(cloudcenters, 255.0, 0.0, 0.0), "cloudcenters");
    cabledetection.viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 13, "cloudcenters");
    
    std::vector<CableDetectionPointNormal::Cable> cables;
    {
        pcl::ScopeTime t("findCables");
        pcl::console::setVerbosityLevel (pcl::console::L_INFO);
        cabledetection.findCables(cables);
        pcl::console::setVerbosityLevel (pcl::console::L_DEBUG);
        size_t terminalindex = 0;
        for (std::vector<CableDetectionPointNormal::Cable>::iterator cableitr = cables.begin(); cableitr != cables.end(); cableitr++){
            //cabledetection.findCableTerminal(*cableitr, 0.018);
            double offset = 0.018;
            //pcl::ModelCoefficients::Ptr terminalcoeffs(new pcl::ModelCoefficients);
            //terminalcoeffs->values.resize(9);
            CableDetectionPointNormal::Cable& cable = *cableitr;
            pcl::console::print_highlight("cable: %d\n", cableitr-cables.begin());
            if (cable.size() < 3) {
                pcl::console::print_highlight("no enough cable slices to find terminals. continue.\n");
                continue;
            }
            // one side
            pcl::ModelCoefficients::Ptr terminalcoeffs(new pcl::ModelCoefficients(*(cabledetection.terminalcylindercoeffs_)));
            Eigen::Vector3f dir; 
            dir(0) = (*cable.begin())->cylindercoeffs->values[3] + (*boost::next(cable.begin(),1))->cylindercoeffs->values[3] + (*boost::next(cable.begin(),2))->cylindercoeffs->values[3];
            dir(1) = (*cable.begin())->cylindercoeffs->values[4] + (*boost::next(cable.begin(),1))->cylindercoeffs->values[4] + (*boost::next(cable.begin(),2))->cylindercoeffs->values[4];
            dir(2) = (*cable.begin())->cylindercoeffs->values[5] + (*boost::next(cable.begin(),1))->cylindercoeffs->values[5] + (*boost::next(cable.begin(),2))->cylindercoeffs->values[5];
            dir.normalize();
            terminalcoeffs->values[0] = (*cable.begin())->cylindercoeffs->values[0] + dir(0) * (offset);
            terminalcoeffs->values[1] = (*cable.begin())->cylindercoeffs->values[1] + dir(1) * (offset);
            terminalcoeffs->values[2] = (*cable.begin())->cylindercoeffs->values[2] + dir(2) * (offset);
            terminalcoeffs->values[3] = dir[0];
            terminalcoeffs->values[4] = dir[1];
            terminalcoeffs->values[5] = dir[2];
            terminalcoeffs->values[6] += 0.005;
            terminalcoeffs->values[7] += 0.040;
            terminalcoeffs->values[8] -= 0.005;
            pcl::PointIndices::Ptr indices(new pcl::PointIndices);
            indices->indices.resize(0);
            FindPointIndicesInsideCylinder<pcl::PointWithRange>(cloudcenters, terminalcoeffs, indices);
            if (indices->indices.size() != 1) {
                pcl::console::print_highlight("could not find good center point from image processing. continue (%d)\n", indices->indices.size());
                // visualization for debugging
                pcl::PointCloud<pcl::PointNormal>::Ptr terminalscenepoints(new pcl::PointCloud<pcl::PointNormal>); 
                pcl::PointIndices::Ptr terminalscenepointsindices(new pcl::PointIndices);
                size_t points = cabledetection._findScenePointIndicesInsideCylinder(terminalcoeffs, terminalscenepointsindices);
                pcl::ExtractIndices<pcl::PointNormal> extract;
                extract.setInputCloud (cabledetection.input_);
                extract.setIndices (terminalscenepointsindices);
                //extract.setNegative (true);
                extract.filter (*terminalscenepoints);
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> rgbfield(terminalscenepoints, 255, 0, 255);
                cabledetection.viewer_->removePointCloud ("terminalpoints_rejected" + terminalindex);
                cabledetection.viewer_->addPointCloud<pcl::PointNormal> (terminalscenepoints, rgbfield,  "terminalpoints_rejected" + terminalindex);
                cabledetection.viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "terminalpoints_rejected" + terminalindex);
                //cabledetection.viewer_->addCylinder(*terminalcoeffs, "cylinder " + terminalindex);
            } else {
                terminalcoeffs->values[0] = cloudcenters->points[indices->indices[0]].x;
                terminalcoeffs->values[1] = cloudcenters->points[indices->indices[0]].y;
                terminalcoeffs->values[2] = cloudcenters->points[indices->indices[0]].z;
                Eigen::Vector3f newdir;
                newdir[0] = cloudcenters->points[indices->indices[0]].x - (*cable.begin())->cylindercoeffs->values[0];
                newdir[1] = cloudcenters->points[indices->indices[0]].y - (*cable.begin())->cylindercoeffs->values[1];
                newdir[2] = cloudcenters->points[indices->indices[0]].z - (*cable.begin())->cylindercoeffs->values[2];
                newdir.normalize();
                terminalcoeffs->values[3] = newdir[0]; 
                terminalcoeffs->values[4] = newdir[1]; 
                terminalcoeffs->values[5] = newdir[2]; 

                cabledetection._estimateTerminalFromInitialCoeffes(terminalcoeffs, boost::lexical_cast<std::string>(terminalindex));
            }
            terminalindex++;

            // the other side
            terminalcoeffs.reset(new pcl::ModelCoefficients(*(cabledetection.terminalcylindercoeffs_)));
            dir(0) = (*boost::prior(cable.end()))->cylindercoeffs->values[3] + (*boost::prior(cable.end(),2))->cylindercoeffs->values[3] + (*boost::prior(cable.end(),3))->cylindercoeffs->values[3];
            dir(1) = (*boost::prior(cable.end()))->cylindercoeffs->values[4] + (*boost::prior(cable.end(),2))->cylindercoeffs->values[4] + (*boost::prior(cable.end(),3))->cylindercoeffs->values[4];
            dir(2) = (*boost::prior(cable.end()))->cylindercoeffs->values[5] + (*boost::prior(cable.end(),2))->cylindercoeffs->values[5] + (*boost::prior(cable.end(),3))->cylindercoeffs->values[5];
            dir.normalize();
            dir *= -1;
            terminalcoeffs->values[0] = (*boost::prior(cable.end()))->cylindercoeffs->values[0] + dir(0) * (offset);
            terminalcoeffs->values[1] = (*boost::prior(cable.end()))->cylindercoeffs->values[1] + dir(1) * (offset);
            terminalcoeffs->values[2] = (*boost::prior(cable.end()))->cylindercoeffs->values[2] + dir(2) * (offset);
            terminalcoeffs->values[3] = dir[0];
            terminalcoeffs->values[4] = dir[1];
            terminalcoeffs->values[5] = dir[2];
            terminalcoeffs->values[6] += 0.005;
            terminalcoeffs->values[7] += 0.040;
            terminalcoeffs->values[8] -= 0.005;
            indices->indices.resize(0);
            FindPointIndicesInsideCylinder<pcl::PointWithRange>(cloudcenters, terminalcoeffs, indices);
            if (indices->indices.size() != 1) {
                pcl::console::print_highlight("could not find good center point from image processing. continue (%d)\n", indices->indices.size());
                // visualization for debugging
                pcl::PointCloud<pcl::PointNormal>::Ptr terminalscenepoints(new pcl::PointCloud<pcl::PointNormal>); 
                pcl::PointIndices::Ptr terminalscenepointsindices(new pcl::PointIndices);
                size_t points = cabledetection._findScenePointIndicesInsideCylinder(terminalcoeffs, terminalscenepointsindices);
                pcl::ExtractIndices<pcl::PointNormal> extract;
                extract.setInputCloud (cabledetection.input_);
                extract.setIndices (terminalscenepointsindices);
                //extract.setNegative (true);
                extract.filter (*terminalscenepoints);
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> rgbfield(terminalscenepoints, 255, 0, 255);
                cabledetection.viewer_->removePointCloud ("terminalpoints_rejected" + terminalindex);
                cabledetection.viewer_->addPointCloud<pcl::PointNormal> (terminalscenepoints, rgbfield,  "terminalpoints_rejected" + terminalindex);
                cabledetection.viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "terminalpoints_rejected" + terminalindex);
                //cabledetection.viewer_->addCylinder(*terminalcoeffs, "cylinder " + terminalindex);
            } else {
                terminalcoeffs->values[0] = cloudcenters->points[indices->indices[0]].x;
                terminalcoeffs->values[1] = cloudcenters->points[indices->indices[0]].y;
                terminalcoeffs->values[2] = cloudcenters->points[indices->indices[0]].z;
                Eigen::Vector3f newdir;
                newdir[0] = cloudcenters->points[indices->indices[0]].x - (*boost::prior(cable.end()))->cylindercoeffs->values[0];
                newdir[1] = cloudcenters->points[indices->indices[0]].y - (*boost::prior(cable.end()))->cylindercoeffs->values[1];
                newdir[2] = cloudcenters->points[indices->indices[0]].z - (*boost::prior(cable.end()))->cylindercoeffs->values[2];
                newdir.normalize();
                terminalcoeffs->values[3] = newdir[0]; 
                terminalcoeffs->values[4] = newdir[1]; 
                terminalcoeffs->values[5] = newdir[2]; 

                cabledetection._estimateTerminalFromInitialCoeffes(terminalcoeffs, boost::lexical_cast<std::string>(terminalindex));
            }
            terminalindex++;
        }
    }
    int i = 0;
    for (typename std::vector<CableDetectionPointNormal::Cable>::iterator itr = cables.begin(); itr != cables.end(); itr++ , i++) {
        std::stringstream ss;
        ss << "cable" << i << "_";
        cabledetection.visualizeCable(*itr, ss.str());
    }
    std::cout << "found " << cables.size() << " cables"<< std::endl;

    cabledetection.RunViewerBackGround();

/*
    pcl::console::setVerbosityLevel (pcl::console::L_DEBUG);
    pcl::ModelCoefficients::Ptr terminalcoeffs(new pcl::ModelCoefficients);
    terminalcoeffs->values.resize(9);
    //terminalcoeffs->values[0] = 0.13451;
    //terminalcoeffs->values[1] = 0.0170317;
    //terminalcoeffs->values[2] = 0.908818;
    //terminalcoeffs->values[0] = 0.136;
    //terminalcoeffs->values[1] = 0.004;
    //terminalcoeffs->values[2] = 0.910;

    terminalcoeffs->values[0] = -0.153;
    terminalcoeffs->values[1] = -0.082;
    terminalcoeffs->values[2] = 1.04;

    //terminalcoeffs->values[3] = 0.820915;
    //terminalcoeffs->values[4] = -0.542296;
    //terminalcoeffs->values[5] = 0.178926;
    //terminalcoeffs->values[3] = 0.80492279;
    //terminalcoeffs->values[4] = -0.58831142;
    //terminalcoeffs->values[5] = 0.07738845;

    terminalcoeffs->values[3] = 0;
    terminalcoeffs->values[4] = -0.70710678;
    terminalcoeffs->values[5] = 0.70710678;

    terminalcoeffs->values[6] = 0.0152369;
    terminalcoeffs->values[7] = 0.0173;
    terminalcoeffs->values[8] = -0.0535;
    //terminalcoeffs->values[0] = (-0.0853928);
    //terminalcoeffs->values[1] = (-0.141702);
    //terminalcoeffs->values[2] = (0.804942);
    //terminalcoeffs->values[3] = (0.316745);
    //terminalcoeffs->values[4] = (-0.817544);
    //terminalcoeffs->values[5] = (0.48093);
    //terminalcoeffs->values[6] = (0.0152369);
    //terminalcoeffs->values[7] = (0.0173);
    //terminalcoeffs->values[8] = (-0.02535);
    cabledetection._estimateTerminalFromInitialCoeffes(terminalcoeffs);
*/
    while (true) {
        boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
    }

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
