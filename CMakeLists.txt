cmake_minimum_required(VERSION 3.0.2)
project(osqp_sqp)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS)
find_package(Eigen3 REQUIRED COMPONENTS)
find_package(OsqpEigen REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message(STATUS "OpenMP found.")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

catkin_package(
 INCLUDE_DIRS include
 DEPENDS EIGEN3 OsqpEigen
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${OsqpEigen_INCLUDE_DIRS}
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_executable(sample1 src/sample1.cpp)
target_link_libraries(sample1 ${OsqpEigen_LIBRARIES})

add_executable(sample2 src/sample2.cpp)
target_link_libraries(sample2 ${OsqpEigen_LIBRARIES})
