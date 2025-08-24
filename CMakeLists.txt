# set clang
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

#set(CC "/opt/homebrew/opt/llvm/bin/clang")
#set(CXX "/opt/homebrew/opt/llvm/bin/clang++")

cmake_minimum_required(VERSION 3.27)
project(Freeform)


# Set the compiler to GCC installed by Homebrew
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
# set(CMAKE_CXX_STANDARD 17)
# message("CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
# message("CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")


list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Release)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()




# Libigl
# add_subdirectory(libigl)
include(libigl)


# Enable targets
igl_include(glfw)
igl_include(imgui)

igl_include(stb)
igl_include(copyleft cgal)
igl_include(predicates)
#igl_include(spectra)


# find_package(CGAL REQUIRED)
# include(${CGAL_USE_FILE})
#

#
# if(APPLE)
# #    set(CMAKE_C_COMPILER clang)
# #    set(CMAKE_CXX_COMPILER clang++)

#    set(OpenMP_C_FLAGS "-Xclang -fopenmp -L/opt/homebrew/opt/llvm/lib")
#    set(OpenMP_C_INCLUDE_DIRS "/opt/homebrew/opt/llvm/include")
#    set(OpenMP_C_LIB_NAMES "libomp")
#    set(OpenMP_libomp_LIBRARY "omp")
# #    set(OpenMP_omp_LIBRARY omp)

#    set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -L/opt/homebrew/opt/llvm/lib")
#    set(OpenMP_CXX_INCLUDE_DIRS "/opt/homebrew/opt/llvm/include")
#    set(OpenMP_CXX_LIB_NAMES "libomp")
#    set(OpenMP_CXX_LIB_NAMES "libomp")
#    set(OpenMP_omp_LIBRARY omp)

# endif()
# OpenMP

set(OpenMP_C_FLAGS "-fopenmp")
set(OpenMP_C_LIB_NAMES "omp")
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/llvm/lib/libomp.dylib")

# Add the OpenMP flags to the compiler options
add_compile_options(${OpenMP_C_FLAGS})
add_compile_options(${OpenMP_CXX_FLAGS})
find_package(OpenMP)

set(AUTODIFF_BUILD_TESTS OFF)
set(AUTODIFF_BUILD_PYTHON OFF)
set(AUTODIFF_BUILD_EXAMPLES OFF)
set(AUTODIFF_BUILD_DOCS OFF)
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "Disable testing for dependencies" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "Disable Eigen testing" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "Disable Eigen unsupported testing" FORCE)
set(AUTODIFF_BUILD_DOCS OFF CACHE BOOL "Disable Eigen unsupported testing" FORCE)

find_package(autodiff REQUIRED)

# find_package(nlohmann_json 3.11.3 REQUIRED)

#Release mode
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add your project files
#file(GLOB SRC_FILES *.cpp)
add_executable(${PROJECT_NAME} main.cpp
        Constraint.cpp
        Constraint.h
        SelfCollision.cpp
        SelfCollision.h
        Optimization.cpp
        Optimization.h
        Isometry.cpp
        Isometry.h
        Closeness.cpp
        Closeness.h
        Boundary.cpp
        Boundary.h
        EdgeFlip.cpp
        EdgeFlip.h
        EdgeCollapse.cpp
        EdgeCollapse.h
        SymmetricDirichlet.cpp
        SymmetricDirichlet.h
        Adam.h
        Angle.cpp
        Angle.h
        Angle_sub.cpp
        Angle_sub.h
        Uniform.cpp
        Uniform.h
        LocalICP.cpp
        LocalICP.h
        data.h)
        # Isometry.cpp
        # Isometry.h
        # Closeness.cpp
        # Closeness.h
        # Boundary.cpp
        # Boundary.h
        # Angle.cpp
        # Angle.h
        # Visualization.cpp
        # Visualization.h
        # ControlPoint.h
        # ControlPoint.cpp
        # Twist.cpp
        # Twist.h
        # TwistConstraint.cpp
        # TwistConstraint.h)
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw igl::imgui igl::glfw igl::stb igl_copyleft::cgal igl::predicates OpenMP::OpenMP_CXX autodiff::autodiff)
# target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/shapeop/libShapeOp/src/)
# target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/shapeop/libShapeOp/api/)

# add_executable(viz visualization/VisualizeResult.cpp
# visualization/angle.cpp
# visualization/angle.h
# visualization/visualization.cpp
# visualization/visualization.h)
# target_link_libraries(viz PUBLIC igl::glfw igl::imgui igl::embree)
message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

add_executable(visualization Visualization.cpp
ConnectFold.cpp
ConnectFold.h
Constraint.cpp
        Constraint.h
        Optimization.cpp
        Optimization.h
        Isometry.cpp
        Isometry.h
        Closeness.cpp
        Closeness.h
        Boundary.cpp
        Boundary.h
        EdgeFlip.cpp
        EdgeFlip.h
        EdgeCollapse.cpp
        EdgeCollapse.h
        SymmetricDirichlet.cpp
        SymmetricDirichlet.h
        Adam.h
        Angle.cpp
        Angle.h
        Angle_sub.cpp
        Angle_sub.h
        Uniform.cpp
        Uniform.h
        LocalICP.cpp
        LocalICP.h
        data.h)
target_link_libraries(visualization PUBLIC igl::glfw igl::imgui igl::glfw igl::stb igl_copyleft::cgal igl::predicates OpenMP::OpenMP_CXX autodiff::autodiff)

add_executable(collision
        test_collision.cpp
        SelfCollision.cpp
        SelfCollision.h
        data.h)
target_link_libraries(collision PUBLIC igl::glfw igl::imgui igl::glfw igl::stb igl_copyleft::cgal igl::predicates OpenMP::OpenMP_CXX autodiff::autodiff)