
cmake_minimum_required(VERSION 3.20)

project(diff_surfel_rasterization LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

find_package(Python3 REQUIRED COMPONENTS Development)


add_library(_C
    MODULE
	cuda_rasterizer/backward.h
	cuda_rasterizer/backward.cu
	cuda_rasterizer/forward.h
	cuda_rasterizer/forward.cu
	cuda_rasterizer/auxiliary.h
	cuda_rasterizer/rasterizer_impl.cu
	cuda_rasterizer/rasterizer_impl.h
	cuda_rasterizer/rasterizer.h
)


set_target_properties(_C PROPERTIES CUDA_ARCHITECTURES "70;75;86")

set_target_properties(_C PROPERTIES
    PREFIX ""
    SUFFIX ".cpython-38-x86_64-linux-gnu.so"  # 根据你的实际需要修改 Python 版本和平台信息
)

target_include_directories(_C PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(_C PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# 将 Python 的包含目录添加到编译器的包含路径中
