cmake_minimum_required(VERSION 3.8)

project("PhotonMapping")

include(cmake/check-compiler.cmake)

set(CMAKE_CXX_STANDARD 17)

set(
	SOURCE_FILES 
		main.cpp
		source/light.cpp
		source/camera.cpp
		source/object.cpp
		source/kdtree_object.cpp
		source/shader.cpp
		source/kdtree_shader.cpp
		source/renderer.cpp
		source/renderer_kdtree.cpp
		source/photon_map.cpp
)

configure_file(include/project_constants.h.in ${PROJECT_BINARY_DIR}/project_constants.h @ONLY)

include_directories("include")
include(cmake/add-libraries-linux.cmake)

add_executable(PhotonMapping ${SOURCE_FILES})

include(cmake/target-link-libraries-linux.cmake)

target_include_directories(PhotonMapping PUBLIC ${CMAKE_BINARY_DIR})