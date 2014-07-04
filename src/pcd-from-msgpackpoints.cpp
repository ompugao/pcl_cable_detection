#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <msgpack.hpp>
#include "msgpack_extension/pcl.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

int main(int argc, char **argv) {
    namespace bpo = boost::program_options;
    namespace bfs = boost::filesystem;
    bpo::options_description opts_desc("Allowed options");
    bpo::positional_options_description p;

    opts_desc.add_options()
        ("input_file", bpo::value< std::string >(), "Input data.")
        ("output_file", bpo::value< std::string >(), "Output pcd")
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
    std::string output_file = opts["output_file"].as<std::string> ();

    // Load file
    std::ifstream fin( input_file.c_str(), std::ios::in | std::ios::binary );
    if (!fin) {
        std::cout << "no such a file: " << input_file;
        return 1;
    }

    // This is target object.
    pcl::PointCloud<pcl::PointXYZ> points;
    
    std::string content( (std::istreambuf_iterator<char>(fin) ),
            (std::istreambuf_iterator<char>()    ) );
    // Deserialize the serialized data.
    msgpack::unpacked msg;    // includes memory pool and deserialized object
    msgpack::unpack(&msg, content.data(), content.size());
    msgpack::object obj = msg.get();

    // Print the deserialized object to stdout.
    //std::cout << obj << std::endl;    // ["Hello," "World!"]

    // Convert the deserialized object to staticaly typed object.
    obj.convert(&points);

    pcl::io::savePCDFileBinaryCompressed (output_file, points);
}
