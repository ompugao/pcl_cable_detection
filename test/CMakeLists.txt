macro(build_sample name)
  add_executable(${name} ${name}.cpp)
  add_dependencies(${name} libpcl-cable-detection)
  #target_link_libraries (${name} libpcl-cable-detection ${PCL_LIBRARIES} ${MSGPACK_LIBRARIES} boost_program_options)
  target_link_libraries (${name} libpcl-cable-detection ${PCL_LIBRARIES} boost_program_options)
  install(TARGETS ${name} DESTINATION bin)
endmacro(build_sample)

build_sample(pcl-cable-detection-test)
#if(MSGPACK_FOUND)
    #build_sample(pcd-from-msgpackpoints)
    #build_sample(pcdconverter)
#endif()
