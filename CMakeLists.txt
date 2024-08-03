cmake_minimum_required(VERSION 3.0.2)
project(osqp_sqp)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS)
find_package(Eigen3 REQUIRED COMPONENTS)
find_package(OsqpEigen REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES osqp_sqp
 DEPENDS EIGEN3 OsqpEigen
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(osqp_sqp
  src/osqp_sqp.cpp
  )
target_link_libraries(osqp_sqp ${OsqpEigen_LIBRARIES})

add_executable(sample1 src/sample1.cpp)
target_link_libraries(sample1 osqp_sqp ${OsqpEigen_LIBRARIES})
