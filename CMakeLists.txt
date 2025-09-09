# set clang
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

cmake_minimum_required(VERSION 3.27)
project(FLE)


set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")


set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Release)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()




# Libigl
add_subdirectory(libigl_fold)
# include(libigl)


# Enable targets
igl_include(glfw)
igl_include(imgui)

igl_include(stb)
igl_include(copyleft cgal)
igl_include(predicates)

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

add_executable(${PROJECT_NAME} src/main.cpp
        src/Constraint.cpp
        src/Constraint.h
        src/SelfCollision.cpp
        src/SelfCollision.h
        src/Optimization.cpp
        src/Optimization.h
        src/Isometry.cpp
        src/Isometry.h
        src/Closeness.cpp
        src/Closeness.h
        src/Boundary.cpp
        src/Boundary.h
        src/EdgeFlip.cpp
        src/EdgeFlip.h
        src/EdgeCollapse.cpp
        src/EdgeCollapse.h
        src/SymmetricDirichlet.cpp
        src/SymmetricDirichlet.h
        src/Adam.h
        src/Angle_sub.cpp
        src/Angle_sub.h
        src/Uniform.cpp
        src/Uniform.h
        src/data.h)
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw igl::imgui igl::glfw igl::stb igl_copyleft::cgal igl::predicates OpenMP::OpenMP_CXX autodiff::autodiff)


add_executable(visualization src/Visualization.cpp
        src/ConnectFold.cpp
        src/ConnectFold.h
        src/Constraint.cpp
        src/Constraint.h
        src/SelfCollision.cpp
        src/SelfCollision.h
        src/Optimization.cpp
        src/Optimization.h
        src/Isometry.cpp
        src/Isometry.h
        src/Closeness.cpp
        src/Closeness.h
        src/Boundary.cpp
        src/Boundary.h
        src/EdgeFlip.cpp
        src/EdgeFlip.h
        src/EdgeCollapse.cpp
        src/EdgeCollapse.h
        src/SymmetricDirichlet.cpp
        src/SymmetricDirichlet.h
        src/Adam.h
        src/Angle_sub.cpp
        src/Angle_sub.h
        src/Uniform.cpp
        src/Uniform.h
        src/data.h)
target_link_libraries(visualization PUBLIC igl::glfw igl::imgui igl::glfw igl::stb igl_copyleft::cgal igl::predicates OpenMP::OpenMP_CXX autodiff::autodiff)

add_executable(collision
        src/test_collision.cpp
        src/SelfCollision.cpp
        src/SelfCollision.h
        src/data.h)
target_link_libraries(collision PUBLIC igl::glfw igl::imgui igl::glfw igl::stb igl_copyleft::cgal igl::predicates OpenMP::OpenMP_CXX autodiff::autodiff)