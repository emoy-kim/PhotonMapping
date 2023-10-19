cmake_minimum_required(VERSION 3.8)

project("PhotonMapping")

include(cmake/check-compiler.cmake)

set(CMAKE_CXX_STANDARD 17)

set(
	SOURCE_FILES 
		main.cpp
		source/camera.cpp
		source/object.cpp
		source/light_object.cpp
		source/kdtree.cpp
		source/shader.cpp
		source/photon_map.cpp
		source/renderer.cpp
		source/renderer_kdtree.cpp
		source/renderer_photon.cpp
)

configure_file(include/project_constants.h.in ${PROJECT_BINARY_DIR}/project_constants.h @ONLY)

find_package(CUDA)

include_directories("include")
include(cmake/add-libraries-linux.cmake)

if(CUDA_FOUND)
	set(
		CUDA_SOURCE_FILES
			source/cuda/kdtree.cu
			source/cuda/photon_map.cu
	)
   cuda_add_executable(PhotonMapping ${SOURCE_FILES} ${CUDA_SOURCE_FILES})
	target_compile_definitions(PhotonMapping PRIVATE USE_CUDA)
else()
	add_executable(PhotonMapping ${SOURCE_FILES})
endif()

include(cmake/target-link-libraries-linux.cmake)

target_include_directories(PhotonMapping PUBLIC ${CMAKE_BINARY_DIR})