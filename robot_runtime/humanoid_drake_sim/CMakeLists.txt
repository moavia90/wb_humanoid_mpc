cmake_minimum_required(VERSION 3.8)
project(humanoid_drake_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 20)
# NOTE: Do NOT set BOOST_MPL_LIMIT_LIST_SIZE globally — it breaks boost/multi_index
# included transitively from humanoid_centroidal_mpc headers.
# PINOCCHIO defines are propagated automatically via pinocchio cmake targets.

add_definitions("-DBOOST_MPL_LIMIT_VECTOR_SIZE=30")
add_definitions("-DBOOST_MPL_CFG_NO_PREPROCESSED_HEADERS")

add_definitions("-DPINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR")
add_definitions("-DPINOCCHIO_URDFDOM_USE_STD_SHARED_PTR")

# Boost MPL limits (required for humanoid_common_mpc headers)
add_definitions("-DBOOST_MPL_LIMIT_LIST_SIZE=30")
add_definitions("-DBOOST_MPL_LIMIT_VECTOR_SIZE=30")
add_definitions("-DBOOST_MPL_CFG_NO_PREPROCESSED_HEADERS")

# PINOCCHIO defines are propagated automatically via pinocchio cmake targets.
add_definitions("-DPINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR")
add_definitions("-DPINOCCHIO_URDFDOM_USE_STD_SHARED_PTR")

find_package(ocs2_sqp REQUIRED)
find_package(robot_model REQUIRED)

# ── Dependencies ──────────────────────────────────────────────────────────────

find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time chrono timer log log_setup)
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# OCS2 packages — all installed as standalone ament packages
find_package(ocs2_ros2_msgs REQUIRED)
find_package(humanoid_mpc_msgs REQUIRED)
find_package(humanoid_common_mpc_ros2 REQUIRED)
find_package(ocs2_mpc REQUIRED)
find_package(ocs2_pinocchio_interface REQUIRED)
find_package(ocs2_centroidal_model REQUIRED)   # in lib/ocs2_ros2/ocs2_pinocchio/
find_package(humanoid_common_mpc REQUIRED)
find_package(humanoid_centroidal_mpc REQUIRED) # in humanoid_nmpc/

# Pinocchio — use find_package (cmake config) which ocs2_centroidal_model requires
find_package(pinocchio REQUIRED)

# Drake (standalone, no drake_ros)
find_package(drake REQUIRED)
find_package(absl REQUIRED)

# ── Include dirs ──────────────────────────────────────────────────────────────

# Boost and Pinocchio includes are safe globally
include_directories(
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${pinocchio_INCLUDE_DIRS}
)

link_directories(${pinocchio_LIBRARY_DIRS})

# ── Node 1: Drake Sim Node ────────────────────────────────────────────────────

add_executable(drake_sim_node src/drake_sim_node.cpp)

# Drake::drake provides all necessary includes and libraries
target_link_libraries(drake_sim_node drake::drake)

ament_target_dependencies(drake_sim_node
  rclcpp
  sensor_msgs
  ament_index_cpp
  ocs2_ros2_msgs
)

# ── Node 2: Policy Bridge Node ────────────────────────────────────────────────

add_executable(policy_to_command_node src/policy_to_command_node.cpp)

target_link_libraries(policy_to_command_node
  pinocchio::pinocchio
  ${Boost_LIBRARIES}
)

ament_target_dependencies(policy_to_command_node
  rclcpp
  sensor_msgs
  std_msgs
  ament_index_cpp
  ocs2_ros2_msgs
  ocs2_mpc
  ocs2_pinocchio_interface
  ocs2_centroidal_model
  humanoid_common_mpc
  humanoid_centroidal_mpc
)

# ── Node 3: Drake MRT Sim Node ────────────────────────────────────────────────

add_executable(drake_mrt_sim_node src/drake_mrt_sim_node.cpp)

# All non‑ament libraries: Pinocchio, Boost, robot_model, Drake, absl
target_link_libraries(drake_mrt_sim_node
  pinocchio::pinocchio
  ${Boost_LIBRARIES}
  drake::drake
  absl::base
  absl::strings
  absl::hash
  absl::raw_hash_set
  absl::flat_hash_map
  absl::throw_delegate
  absl::city
  absl::low_level_hash
  /home/mov/Documents/humanoid_mpc_ws/install/robot_model/lib/librobot_model.a
)

# All ament / ROS 2 package names
ament_target_dependencies(drake_mrt_sim_node
  rclcpp
  sensor_msgs
  std_msgs
  ament_index_cpp
  ocs2_sqp
  ocs2_mpc
  ocs2_pinocchio_interface
  ocs2_centroidal_model
  humanoid_common_mpc
  humanoid_common_mpc_ros2
  humanoid_centroidal_mpc
)

# ── Install ───────────────────────────────────────────────────────────────────

install(TARGETS
  drake_sim_node
  policy_to_command_node
  drake_mrt_sim_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()