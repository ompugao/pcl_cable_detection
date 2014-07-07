# -*- coding: utf-8 -*-
include_directories(${PCL_CABLE_DETECTION_INCLUDE_LOCAL_DIRS})
link_directories(${PCL_CABLE_DETECTION_LINK_DIRS})

set(LIBRARY_SRCFILES pcl-cable-detection.cpp)

add_library(libpcl-cable-detection SHARED ${LIBRARY_SRCFILES})
if( EXTRA_MSVC_DEPEND )
  message(STATUS "adding msvc_boost dependency")
  add_dependencies(libpcl-cable-detection ${EXTRA_MSVC_DEPEND})
endif()

set_target_properties(libpcl-cable-detection PROPERTIES COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} ${Boost_CFLAGS} -DPCL_CABLE_DETECTION_DLL_EXPORTS -DPCL_CABLE_DETECTION_DLL" LINK_FLAGS ""
  OUTPUT_NAME pcl-cable-detection${PCL_CABLE_DETECTION_LIBRARY_SUFFIX}
  SOVERSION 0 # always have it 0 since we're including the soversion as part of the library name
  VERSION ${PCL_CABLE_DETECTION_VERSION}
  CLEAN_DIRECT_OUTPUT 1
  )
target_link_libraries(libpcl-cable-detection ${PCL_LIBRARIES} ${EXTRA_LIBRARIES})
if( MSVC )
  install(TARGETS libpcl-cable-detection RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib${LIB_SUFFIX})
else()
  install(TARGETS libpcl-cable-detection DESTINATION lib${LIB_SUFFIX})
endif()
