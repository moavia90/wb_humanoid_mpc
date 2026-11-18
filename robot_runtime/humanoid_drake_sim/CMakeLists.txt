cmake_minimum_required(VERSION 3.8)
project(humanoid_drake_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 20)
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
find_package(ocs2_ros2_interfaces REQUIRED)
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
find_package(urdf REQUIRED)

# ── Include dirs ──────────────────────────────────────────────────────────────

# Boost and Pinocchio includes are safe globally
include_directories(
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${pinocchio_INCLUDE_DIRS}
)

link_directories(${pinocchio_LIBRARY_DIRS})

# ── Physics backend: RobotHWInterfaceBase implementation ─────────────────────

add_library(drake_sim_interface src/DrakeSimInterface.cpp)
target_link_libraries(drake_sim_interface
  drake::drake
  ${urdf_LIBRARIES}
  absl::base
  absl::strings
  absl::hash
  absl::raw_hash_set
  absl::flat_hash_map
  absl::throw_delegate
  absl::city
  absl::low_level_hash
)
target_include_directories(drake_sim_interface
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(drake_sim_interface
  ament_index_cpp
  robot_model
)

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

add_executable(drake_urdf_interface_smoke_test
  src/drake_urdf_interface_smoke_test.cpp
)
target_link_libraries(drake_urdf_interface_smoke_test drake_sim_interface)
ament_target_dependencies(drake_urdf_interface_smoke_test robot_model)

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

# The MRT executable uses the reusable physics-only RobotHWInterface backend.
target_link_libraries(drake_mrt_sim_node
  drake_sim_interface
  pinocchio::pinocchio
  ${Boost_LIBRARIES}
  absl::base
  absl::strings
  absl::hash
  absl::raw_hash_set
  absl::flat_hash_map
  absl::throw_delegate
  absl::city
  absl::low_level_hash
)

# All ament / ROS 2 package names
ament_target_dependencies(drake_mrt_sim_node
  rclcpp
  sensor_msgs
  std_msgs
  ament_index_cpp
  ocs2_ros2_interfaces
  ocs2_ros2_msgs
  ocs2_mpc
  ocs2_pinocchio_interface
  ocs2_centroidal_model
  humanoid_mpc_msgs
  humanoid_common_mpc
  humanoid_common_mpc_ros2
  humanoid_centroidal_mpc
  robot_model
)

# ── Legacy SAP-PD diagnostic (not used by drake_mrt_sim_node) ────────────────

add_executable(drake_sap_stand_test src/drake_sap_stand_test.cpp)

target_link_libraries(drake_sap_stand_test
  drake::drake
)

ament_target_dependencies(drake_sap_stand_test
  ament_index_cpp
)
# ──────────────────────────────────────────────────────────────────────────────
# Legacy MPC policy experiments. The production URDF backend is validated by
# drake_urdf_interface_smoke_test above.
# ──────────────────────────────────────────────────────────────────────────────

# Stage 2: static equilibrium gate. Pure Drake — no OCS2, no robot_model.
add_executable(static_equilibrium_check src/static_equilibrium_check.cpp)
target_link_libraries(static_equilibrium_check drake::drake)
ament_target_dependencies(static_equilibrium_check ament_index_cpp)

# Stage 3: gravity-comp hold. Pure Drake.
add_executable(gravity_comp_hold_test src/gravity_comp_hold_test.cpp)
target_link_libraries(gravity_comp_hold_test drake::drake)
ament_target_dependencies(gravity_comp_hold_test ament_index_cpp)

# Stage 1: kinematic replay. OCS2 (solve once) + Drake (display).
# No robot_model -> no librobot_model.a needed; absl components kept in case
# OCS2/Drake headers pull absl symbols (harmless if unused).
add_executable(kinematic_replay_test src/kinematic_replay_test.cpp)
target_link_libraries(kinematic_replay_test
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
)
ament_target_dependencies(kinematic_replay_test
  rclcpp
  ament_index_cpp
  ocs2_sqp
  ocs2_mpc
  ocs2_pinocchio_interface
  ocs2_centroidal_model
  humanoid_common_mpc
  humanoid_centroidal_mpc
)

# Stage 4: open-loop policy playback. Same recipe as Stage 1.
add_executable(policy_playback_test src/policy_playback_test.cpp)
target_link_libraries(policy_playback_test
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
)
ament_target_dependencies(policy_playback_test
  rclcpp
  ament_index_cpp
  ocs2_sqp
  ocs2_mpc
  ocs2_pinocchio_interface
  ocs2_centroidal_model
  humanoid_common_mpc
  humanoid_centroidal_mpc
)

# Stage 5: threaded closed loop.
add_executable(threaded_closed_loop_test src/threaded_closed_loop_test.cpp)
target_link_libraries(threaded_closed_loop_test
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
)
ament_target_dependencies(threaded_closed_loop_test
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
  robot_model
)


# ── Install ───────────────────────────────────────────────────────────────────

install(TARGETS drake_sim_interface
  EXPORT export_drake_sim_interface
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY include/${PROJECT_NAME}
  DESTINATION include
)

install(TARGETS
  drake_sim_node
  drake_urdf_interface_smoke_test
  policy_to_command_node
  drake_mrt_sim_node
  drake_sap_stand_test
  static_equilibrium_check
  gravity_comp_hold_test
  kinematic_replay_test
  policy_playback_test
  threaded_closed_loop_test
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_targets(export_drake_sim_interface HAS_LIBRARY_TARGET)
ament_export_dependencies(
  absl
  ament_index_cpp
  drake
  robot_model
  urdf
)
ament_package()
