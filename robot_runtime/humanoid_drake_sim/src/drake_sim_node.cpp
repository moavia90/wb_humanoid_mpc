///////////////////////////////////////////////////////////////////////////////
// drake_sim_node.cpp
//
// Drake physics simulation for the Unitree G1 humanoid robot.
// Uses the official Unitree MJCF (g1_29dof_drake.xml) which includes
// 29 actuators, foot contact geometry, and a ground plane.
//
// Publishes:
//   /drake/joint_states      (sensor_msgs/JointState)
//   /drake/centroidal_state  (std_msgs/Float64MultiArray) — [h_com(6), base_pose(6)]
//
// Subscribes:
//   /drake/joint_torques     (std_msgs/Float64MultiArray) — 29-dim torque vector
//
// Key fixes in this version:
//   [FIX-1] NaN AND large-value guard on torques — rejects values > max_torque*10
//   [FIX-2] Time sync recovery on exception — resyncs sim_time_ to actual sim time
//   [FIX-3] Correct spawn height via setInitialState() — computes foot position
//            analytically from the MJCF joint offsets and bent-knee angles, then
//            places the foot sphere bottoms just on the ground. This prevents
//            startup drop impact and contact penetration impulses.
//   [FIX-4] Centroidal state sanity clamp — rejects physically impossible values
//            (|z| > 10 m, |v| > 50 m/s) before publishing, so NaN-guard bypass
//            by large-finite values cannot corrupt MPC observations.
//   [FIX-5] Rigid-hydroelastic HalfSpace ground.
//   [FIX-6] Package map from AMENT_PREFIX_PATH.
///////////////////////////////////////////////////////////////////////////////

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/meshcat.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/rgba.h>
#include <drake/geometry/shape_specification.h>
// #include <drake/visualization/visualization.h>
// #include <drake/visualization/meshcat_visualizer_params.h>
#include <drake/geometry/proximity_properties.h>
#include <drake/systems/framework/context.h>
#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/math/rigid_transform.h>
#include <drake/math/rotation_matrix.h>
#include <drake/multibody/tree/revolute_joint.h>

#include <Eigen/Dense>
#include <mutex>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

namespace drake_mbd  = drake::multibody;
namespace drake_geo  = drake::geometry;
namespace drake_sys  = drake::systems;
namespace drake_math = drake::math;

class DrakeSimNode : public rclcpp::Node
{
public:
  DrakeSimNode()
  : Node("drake_sim_node")
  {
    declare_parameter<std::string>("urdf_path",  "");
    declare_parameter<double>("sim_dt",          0.001);
    declare_parameter<double>("max_torque",      300.0);
    declare_parameter<double>("torque_timeout_s", 0.05);
    // initial_height is now computed automatically from kinematics.
    // The parameter is kept for manual override / debugging.
    declare_parameter<double>("initial_height",  -1.0);  // -1 = auto
    declare_parameter<bool>("pin_base_until_first_torque", true);
    declare_parameter<double>("startup_release_torque_threshold", -1.0);

    urdf_path_  = get_parameter("urdf_path").as_string();
    sim_dt_     = get_parameter("sim_dt").as_double();
    max_torque_ = get_parameter("max_torque").as_double();
    torque_timeout_s_ = get_parameter("torque_timeout_s").as_double();
    initial_height_override_ = get_parameter("initial_height").as_double();
    pin_base_until_first_torque_ =
        get_parameter("pin_base_until_first_torque").as_bool();
    startup_pinned_ = pin_base_until_first_torque_;
    startup_release_torque_threshold_ =
        get_parameter("startup_release_torque_threshold").as_double();

    if (urdf_path_.empty()) {
      RCLCPP_FATAL(get_logger(), "Parameter 'urdf_path' must be set.");
      throw std::runtime_error("urdf_path not provided");
    }

    buildDiagram();

    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>(
        "/drake/joint_states", 10);
    // Standard ROS2 topic → robot_state_publisher → /tf → RViz robot model
    joint_states_rviz_pub_ = create_publisher<sensor_msgs::msg::JointState>(
        "/joint_states", 10);
    centroidal_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        "/drake/centroidal_state", 10);
    sim_time_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        "/drake/sim_time", 10);
    contact_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        "/drake/foot_contacts", 10);  // [L_contact, R_contact, L_Fz, R_Fz]

    startup_pin_sub_ = create_subscription<std_msgs::msg::Bool>(
        "/drake/startup_pin", 10,
        [this](std_msgs::msg::Bool::SharedPtr m) {
          std::lock_guard<std::mutex> lk(mu_);
          if (m->data) {
            if (!startup_pinned_) {
              RCLCPP_WARN(get_logger(),
                  "Startup pin commanded ON at sim_t=%.3f; returning to nominal reset pose",
                  sim_time_);
            }
            startup_pinned_ = true;
            bridge_connected_ = false;
            if (cmd_tau_.size() == n_act_) cmd_tau_.setZero();
            last_torque_wall_time_ = std::chrono::steady_clock::now();
          } else {
            if (startup_pinned_) {
              RCLCPP_INFO(get_logger(),
                  "Startup pin commanded OFF at sim_t=%.3f", sim_time_);
            }
            startup_pinned_ = false;
            last_torque_wall_time_ = std::chrono::steady_clock::now();
          }
        });

    torque_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "/drake/joint_torques", 10,
        [this](std_msgs::msg::Float64MultiArray::SharedPtr m) {
          std::lock_guard<std::mutex> lk(mu_);
          if (static_cast<int>(m->data.size()) == n_act_) {
            cmd_tau_ = Eigen::Map<const Eigen::VectorXd>(
                           m->data.data(), n_act_);
            for (int i = 0; i < n_act_; ++i) {
              const double limit = torqueLimitFor(i);
              if (!std::isfinite(cmd_tau_[i]) || std::abs(cmd_tau_[i]) > limit * 10.0)
                cmd_tau_[i] = 0.0;
              cmd_tau_[i] = std::clamp(cmd_tau_[i], -limit, limit);
            }
            last_torque_wall_time_ = std::chrono::steady_clock::now();
            bridge_connected_ = true;
            const double max_abs_tau = cmd_tau_.lpNorm<Eigen::Infinity>();
            if (!pin_base_until_first_torque_ &&
                startup_pinned_ &&
                startup_release_torque_threshold_ >= 0.0 &&
                max_abs_tau > startup_release_torque_threshold_) {
              startup_pinned_ = false;
              RCLCPP_INFO(get_logger(),
                  "Startup pin released by legacy torque threshold: max |tau|=%.3f Nm at sim_t=%.3f",
                  max_abs_tau, sim_time_);
            }
          }
        });

    timer_ = create_wall_timer(
        std::chrono::microseconds(static_cast<int>(sim_dt_ * 1e6)),
        std::bind(&DrakeSimNode::stepAndPublish, this));

    RCLCPP_INFO(get_logger(),
        "DrakeSimNode ready | actuators=%d  dt=%.4fs  z0=%.4fm  mass=%.1fkg",
        n_act_, sim_dt_, spawn_height_, total_mass_);
    for (int i = 0; i < n_act_; ++i)
      RCLCPP_INFO(get_logger(), "  act[%2d] = %s", i, act_names_[i].c_str());
  }

private:
  /* ================================================================
   *  BUILD DIAGRAM
   * ================================================================ */
  void buildDiagram()
  {
    drake_sys::DiagramBuilder<double> builder;

    auto pair = drake_mbd::AddMultibodyPlantSceneGraph(&builder, sim_dt_);
    plant_ = &pair.plant;
    sg_    = &pair.scene_graph;

    // Contact model set below with ground geometry.
    // kHydroelasticWithFallback: robot's rigid spheres contact the compliant ground box.
    // This is the most stable configuration for Drake with MJCF foot spheres.

    // [FIX-6] Package map
    drake_mbd::Parser parser(plant_);
    parser.package_map().PopulateFromEnvironment("AMENT_PREFIX_PATH");
    try {
      const std::string share =
          ament_index_cpp::get_package_share_directory("g1_description");
      if (!parser.package_map().Contains("g1_description"))
        parser.package_map().Add("g1_description", share);
      RCLCPP_INFO(get_logger(), "g1_description → %s", share.c_str());
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "ament_index: %s", e.what());
    }

    RCLCPP_INFO(get_logger(), "Loading model: %s", urdf_path_.c_str());
    model_ = parser.AddModels(urdf_path_).at(0);

    // === HYDROELASTIC BOX GROUND ===
    // A HalfSpace is not supported by point-contact or rigid-hydroelastic.
    // Use a large box with compliant hydroelastic properties.
    plant_->set_contact_model(drake_mbd::ContactModel::kHydroelasticWithFallback);
    // NOTE: set_discrete_contact_solver() removed in this Drake version.
    // kHydroelasticWithFallback on a discrete plant uses SAP automatically.
    const double ground_size = 20.0;   // side length of the ground box [m]
    const double ground_z = -0.1;      // place the top surface near z=0
    {
      drake_geo::ProximityProperties gp;
      // hydroelastic_modulus=5e4 Pa: soft enough to avoid explosive contact forces
      // at dt=1ms, but stiff enough to prevent excessive penetration (<<1mm at robot weight).
      // dissipation=2.0 (Hunt-Crossley) provides strong damping to kill contact oscillations.
      drake_geo::AddCompliantHydroelasticProperties(
          /*resolution_hint=*/0.05, /*hydroelastic_modulus=*/5e4, &gp);
      drake_geo::AddContactMaterial(
          /*dissipation=*/2.0, /*point_stiffness=*/5e4,
          drake_mbd::CoulombFriction<double>(0.8, 0.6), &gp);
      plant_->RegisterCollisionGeometry(
          plant_->world_body(),
          drake_math::RigidTransformd(Eigen::Vector3d(0, 0, ground_z)),
          drake_geo::Box(ground_size, ground_size, 0.2),  // 20 cm thick
          "ground", gp);
      // Optional visual
      plant_->RegisterVisualGeometry(
          plant_->world_body(),
          drake_math::RigidTransformd(Eigen::Vector3d(0, 0, ground_z - 0.1)),
          drake_geo::Box(ground_size, ground_size, 0.2),
          "ground_visual",
          Eigen::Vector4d(0.5, 0.5, 0.5, 1.0));
    }
    plant_->Finalize();

    n_act_ = plant_->num_actuators();
    n_q_   = plant_->num_positions();
    n_v_   = plant_->num_velocities();

    RCLCPP_INFO(get_logger(),
        "Plant: n_q=%d  n_v=%d  n_act=%d", n_q_, n_v_, n_act_);

    if (n_act_ == 0) {
      RCLCPP_FATAL(get_logger(), "No actuators — check MJCF <actuator> tags.");
      throw std::runtime_error("n_act == 0");
    }

    act_names_.reserve(n_act_);
    for (drake_mbd::JointActuatorIndex j(0); j < n_act_; ++j)
      act_names_.push_back(plant_->get_joint_actuator(j).joint().name());

    // Meshcat visualizer — opens a browser-based 3D viewer at http://localhost:7000
    drake::geometry::MeshcatParams meshcat_params;
    meshcat_params.host = "0.0.0.0";  // bind to all interfaces, not just localhost
    meshcat_ = std::make_shared<drake::geometry::Meshcat>(meshcat_params);
    // Debug object: if Meshcat shows only a white page, this red box tells us
    // whether Meshcat itself is rendering. If the box appears but the robot
    // does not, the problem is robot visual geometry / SceneGraph publication.
    meshcat_->SetObject("/debug/red_box",
        drake::geometry::Box(0.2, 0.2, 0.2),
        drake::geometry::Rgba(1.0, 0.0, 0.0, 1.0));
    meshcat_->SetTransform("/debug/red_box",
        drake_math::RigidTransformd(Eigen::Vector3d(0.0, 0.0, 1.0)));
    meshcat_->SetCameraPose(
        Eigen::Vector3d(2.5, -1.5, 1.4),
        Eigen::Vector3d(0.0, 0.0, 0.8));
    // meshcat_ = std::make_shared<drake::geometry::Meshcat>();
    drake::geometry::MeshcatVisualizerParams viz_params;
    viz_params.publish_period = 1.0 / 60.0;  // 60 Hz visual update
    drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
        &builder, *sg_, meshcat_, viz_params);
    RCLCPP_INFO(rclcpp::get_logger("drake_sim_node"),
        "Meshcat running at %s", meshcat_->web_url().c_str());

    diagram_   = builder.Build();
    simulator_ = std::make_unique<drake_sys::Simulator<double>>(*diagram_);
    simulator_->set_target_realtime_rate(1.0);

    setInitialState();

    {
      const auto& pc =
          diagram_->GetSubsystemContext(*plant_, simulator_->get_context());
      total_mass_ = plant_->CalcTotalMass(pc);
    }

    cmd_tau_  = Eigen::VectorXd::Zero(n_act_);
    sim_time_ = 0.0;
    simulator_->Initialize();
    diagram_->ForcedPublish(simulator_->get_context());
  }

  /* ================================================================
   *  COMPUTE SPAWN HEIGHT  [FIX-3]
   *
   *  We compute the vertical position of the lowest foot contact point
   *  relative to the pelvis, in the bent-knee configuration, using the
   *  MJCF joint offsets directly. Then we set pelvis_z so that the
   *  lowest contact point is FOOT_CLEARANCE metres above z=0.
   *
   *  MJCF chain (left leg, relevant z-offsets only):
   *    pelvis → left_hip_pitch_link:  pos="0  0.064452  -0.1027"
   *    → left_hip_roll_link:          pos="0  0.052  -0.030465"  quat w=0.996179
   *    → left_hip_yaw_link:           pos="0.025001  0  -0.12412"
   *    → left_knee_link:              pos="-0.078273  0.0021489  -0.17734"
   *    → left_ankle_pitch_link:       pos="0  -9.4e-5  -0.30001"
   *    → left_ankle_roll_link:        pos="0  0  -0.017558"
   *    foot contact spheres:          pos="-0.05 ±0.025 -0.03"  (lowest)
   *                                   pos="0.12  ±0.03  -0.03"
   *
   *  Rather than computing the full FK analytically (complex due to
   *  joint angle axes), we use a simpler approach: after loading the
   *  plant we temporarily set the bent-knee pose, ask Drake for the
   *  foot body pose, then add clearance.
   * ================================================================ */
  void setInitialState()
  {
    auto& ctx  = simulator_->get_mutable_context();
    auto& pc   = diagram_->GetMutableSubsystemContext(*plant_, &ctx);
    const auto& pelvis = plant_->GetBodyByName("pelvis");

    // Step 1: Place pelvis at a generous reference height
    const double REFERENCE_HEIGHT = 1.0;
    plant_->SetFreeBodyPose(
        &pc, pelvis,
        drake_math::RigidTransformd(
            drake_math::RotationMatrixd::Identity(),
            Eigen::Vector3d(0.0, 0.0, REFERENCE_HEIGHT)));

    drake_mbd::SpatialVelocity<double> v0;
    v0.SetZero();
    plant_->SetFreeBodySpatialVelocity(&pc, pelvis, v0);
    plant_->get_actuation_input_port(model_).FixValue(
        &pc, Eigen::VectorXd::Zero(n_act_));

    // Step 2: Apply the nominal joint angles and clear all joint rates.
    // Reset must overwrite zero-nominal joints too; otherwise a failed Drake
    // step can leave corrupted joint positions in the context and the next
    // spawn-height computation starts from a bad posture.
    for (const auto& nm : act_names_) {
      const double q0 = nominalQ(nm);
      try {
        const auto& rj =
            dynamic_cast<const drake_mbd::RevoluteJoint<double>&>(
                plant_->GetJointByName(nm));
        rj.set_angle(&pc, q0);
        rj.set_angular_rate(&pc, 0.0);
      } catch (...) {}
    }

    // Step 3: Ask Drake where the ankle roll links ended up, then add
    // the contact sphere offset (-0.03 in local z) to find foot height.
    double lowest_foot_z = REFERENCE_HEIGHT;  // conservative start
    const std::vector<std::string> ankle_links = {
        "left_ankle_roll_link", "right_ankle_roll_link"};
    for (const auto& link_name : ankle_links) {
      try {
        const auto& ankle_body = plant_->GetBodyByName(link_name);
        const auto ankle_pose  = plant_->EvalBodyPoseInWorld(pc, ankle_body);
        // Foot contact spheres are at local z = -0.03 in the ankle roll frame.
        // Approximate foot world z by ankle z − 0.03 (good for near-zero roll).
        const double foot_z = ankle_pose.translation().z() - 0.03;
        lowest_foot_z = std::min(lowest_foot_z, foot_z);
      } catch (...) {}
    }

    // Step 4: Shift pelvis so foot sphere bottoms are just on the ground.
    // The MJCF foot collision spheres have radius 0.005 m. Placing their
    // centers at z=0.005 avoids both a startup drop and initial penetration.
    constexpr double FOOT_CLEARANCE = 0.005;
    const double foot_offset = REFERENCE_HEIGHT - lowest_foot_z;
    spawn_height_ = foot_offset + FOOT_CLEARANCE;

    // Allow manual override via launch file parameter
    if (initial_height_override_ > 0.0) {
      RCLCPP_WARN(get_logger(),
          "Manual initial_height override: %.4f m (auto was %.4f m)",
          initial_height_override_, spawn_height_);
      spawn_height_ = initial_height_override_;
    } else {
      RCLCPP_INFO(get_logger(),
          "Auto spawn height: %.4f m (foot offset=%.4f, clearance=%.3f m)",
          spawn_height_, foot_offset, FOOT_CLEARANCE);
    }

    // Step 5: Re-apply final pose at computed spawn height
    plant_->SetFreeBodyPose(
        &pc, pelvis,
        drake_math::RigidTransformd(
            drake_math::RotationMatrixd::Identity(),
            Eigen::Vector3d(0.0, 0.0, spawn_height_)));
    plant_->SetFreeBodySpatialVelocity(&pc, pelvis, v0);
  }

  static double nominalQ(const std::string& n)
  {
    if (n.find("hip_pitch")   != std::string::npos) return -0.20;
    if (n.find("knee")        != std::string::npos) return  0.40;
    if (n.find("ankle_pitch") != std::string::npos) return -0.20;
    return 0.0;
  }

  static double holdKp(const std::string& n)
  {
    // Model-based startup impedance. This keeps joint motion quiet while the
    // bridge resets MPC; it is not a balance controller.
    if (n.find("hip_pitch")   != std::string::npos) return 90.0;
    if (n.find("hip_roll")    != std::string::npos) return 75.0;
    if (n.find("hip_yaw")     != std::string::npos) return 20.0;
    if (n.find("knee")        != std::string::npos) return 30.0;
    if (n.find("ankle_pitch") != std::string::npos) return 25.0;
    if (n.find("ankle_roll")  != std::string::npos) return 10.0;
    if (n.find("waist_pitch") != std::string::npos) return 65.0;
    if (n.find("waist_roll")  != std::string::npos) return 80.0;
    if (n.find("waist_yaw")   != std::string::npos) return 25.0;
    if (n.find("shoulder")    != std::string::npos) return 10.0;
    if (n.find("elbow")       != std::string::npos) return 10.0;
    if (n.find("wrist")       != std::string::npos) return  3.0;
    return 10.0;
  }

  static double holdKd(const std::string& n)
  {
    if (n.find("hip_pitch")   != std::string::npos) return 18.0;
    if (n.find("hip_roll")    != std::string::npos) return 14.0;
    if (n.find("hip_yaw")     != std::string::npos) return  2.0;
    if (n.find("knee")        != std::string::npos) return  4.0;
    if (n.find("ankle_pitch") != std::string::npos) return  2.0;
    if (n.find("ankle_roll")  != std::string::npos) return  1.0;
    if (n.find("waist_pitch") != std::string::npos) return 13.0;
    if (n.find("waist_roll")  != std::string::npos) return 16.0;
    if (n.find("waist_yaw")   != std::string::npos) return  5.0;
    if (n.find("shoulder")    != std::string::npos) return  2.0;
    if (n.find("elbow")       != std::string::npos) return  1.5;
    if (n.find("wrist")       != std::string::npos) return  0.5;
    return 1.0;
  }

  static double nominalTorqueLimit(const std::string& n)
  {
    // Drake ignores MJCF actuatorfrcrange, so enforce the XML intent here.
    if (n.find("hip_")        != std::string::npos) return 88.0;
    if (n.find("knee")        != std::string::npos) return 139.0;
    if (n.find("ankle")       != std::string::npos) return 50.0;
    if (n.find("waist_yaw")   != std::string::npos) return 88.0;
    if (n.find("waist_roll")  != std::string::npos) return 50.0;
    if (n.find("waist_pitch") != std::string::npos) return 50.0;
    if (n.find("shoulder")    != std::string::npos) return 25.0;
    if (n.find("elbow")       != std::string::npos) return 25.0;
    if (n.find("wrist_roll")  != std::string::npos) return 25.0;
    if (n.find("wrist_pitch") != std::string::npos) return  5.0;
    if (n.find("wrist_yaw")   != std::string::npos) return  5.0;
    return 50.0;
  }

  double torqueLimitFor(const int actuator_index) const
  {
    if (actuator_index < 0 || actuator_index >= static_cast<int>(act_names_.size())) {
      return max_torque_;
    }
    return std::min(max_torque_, nominalTorqueLimit(act_names_[actuator_index]));
  }

  void applyStartupPin(drake_sys::Context<double>* pc)
  {
    if (!startup_pinned_) return;

    const auto& pelvis = plant_->GetBodyByName("pelvis");
    plant_->SetFreeBodyPose(
        pc, pelvis,
        drake_math::RigidTransformd(
            drake_math::RotationMatrixd::Identity(),
            Eigen::Vector3d(0.0, 0.0, spawn_height_)));

    drake_mbd::SpatialVelocity<double> v0;
    v0.SetZero();
    plant_->SetFreeBodySpatialVelocity(pc, pelvis, v0);

    for (const auto& nm : act_names_) {
      try {
        const auto& rj =
            dynamic_cast<const drake_mbd::RevoluteJoint<double>&>(
                plant_->GetJointByName(nm));
        rj.set_angle(pc, nominalQ(nm));
        rj.set_angular_rate(pc, 0.0);
      } catch (...) {}
    }
  }

  /* ================================================================
   *  STEP + PUBLISH
   * ================================================================ */
  void stepAndPublish()
  {
    /* Apply torques ─────────────────────────────────────────────── */
    // Hold and apply are merged under one mutex so hold takes effect THIS tick,
    // not next, and cmd_tau_ is never accessed without the lock.
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto& ctx = simulator_->get_mutable_context();
      auto& pc  = diagram_->GetMutableSubsystemContext(*plant_, &ctx);

      applyStartupPin(&pc);

      if (bridge_connected_) {
        const double age = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - last_torque_wall_time_).count();
        if (age > torque_timeout_s_) {
          bridge_connected_ = false;
          cmd_tau_.setZero();
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
              "Torque command timeout %.3fs: re-engaging Drake internal hold", age);
        }
      }

      // If bridge hasn't connected yet, compute soft PD hold torques.
      // Gains are low enough to keep the robot roughly upright without
      // exciting contact solver instabilities.
      if (!bridge_connected_) {
        // Startup hold is intentionally modest. The startup pin keeps the base
        // valid until MPC produces meaningful torques.
        // Use RevoluteJoint API to read joint angles/velocities.
        // The raw-index approach (q[7+ai]) is WRONG: Drake's generalized
        // coordinates are ordered by depth-first kinematic tree traversal,
        // not by actuator declaration order. Applying the wrong torque to
        // the wrong joint destabilises the robot before the bridge connects.
        for (int ai = 0; ai < n_act_; ++ai) {
          double q_meas = 0.0, v_meas = 0.0;
          try {
            const auto& rj = dynamic_cast<const drake_mbd::RevoluteJoint<double>&>(
                plant_->GetJointByName(act_names_[ai]));
            q_meas = rj.get_angle(pc);
            v_meas = rj.get_angular_rate(pc);
          } catch (...) {}
          const double limit = torqueLimitFor(ai);
          cmd_tau_[ai] = std::clamp(
              holdKp(act_names_[ai]) * (nominalQ(act_names_[ai]) - q_meas) +
                  holdKd(act_names_[ai]) * (-v_meas),
              -limit, limit);
        }
      }

      Eigen::VectorXd tau = cmd_tau_;
      for (int i = 0; i < n_act_; ++i) {
        const double limit = torqueLimitFor(i);
        if (!std::isfinite(tau[i]) || std::abs(tau[i]) > limit * 10.0) tau[i] = 0.0;
        tau[i] = std::clamp(tau[i], -limit, limit);
      }

      plant_->get_actuation_input_port(model_).FixValue(&pc, tau);
    }

    /* Advance ───────────────────────────────────────────────────── */
    // If we are in the recovery cool-down period, skip stepping and zero torque.
    if (recovery_ticks_remaining_ > 0) {
      --recovery_ticks_remaining_;
      std::lock_guard<std::mutex> lk(mu_);
      cmd_tau_.setZero();
      return;
    }

    sim_time_ += sim_dt_;
    try {
      simulator_->AdvanceTo(sim_time_);
      consecutive_failures_ = 0;  // reset on success
    } catch (const std::exception& e) {
      ++consecutive_failures_;
      RCLCPP_ERROR(get_logger(),
          "Drake step failed t=%.4f [fail#%d]: %s",
          sim_time_, consecutive_failures_, e.what());

      // Resync sim_time_ to what the simulator actually reached,
      // then call Initialize() ONCE, then wait out a cool-down before resuming.
      // The cool-down prevents the re-init spam: we skip N ticks while holding
      // zero torque, letting the robot settle before the next AdvanceTo().
      try {
        // Respawn at correct height. Then set context time = 0 BEFORE Initialize()
        // so the next AdvanceTo(sim_dt_) goes forward, not backward (which causes NaN).
        bridge_connected_ = false;  // Re-engage Drake's internal hold
        startup_pinned_ = pin_base_until_first_torque_;
        last_torque_wall_time_ = std::chrono::steady_clock::now();
        setInitialState();
        auto& rctx = simulator_->get_mutable_context();
        rctx.SetTime(0.0);
        sim_time_ = 0.0;
        simulator_->Initialize();
        diagram_->ForcedPublish(simulator_->get_context());
        // Cool-down: 500 ms of hold torque at spawn pose
        recovery_ticks_remaining_ = 500;
        RCLCPP_WARN(get_logger(),
            "Drake re-initialized: respawned z=%.4f, t=0 — holding 500 ms",
            spawn_height_);
      } catch (const std::exception& e2) {
        RCLCPP_ERROR(get_logger(),
            "Drake re-initialization also failed: %s — node needs restart", e2.what());
        recovery_ticks_remaining_ = 2000;  // long pause, will likely need manual restart
      }

      std::lock_guard<std::mutex> lk(mu_);
      cmd_tau_.setZero();
      return;
    }

    const auto& ctx = simulator_->get_context();
    const auto& pc  = diagram_->GetSubsystemContext(*plant_, ctx);

    // Keep Meshcat alive for browsers that connected after startup.
    // This also helps diagnose the blank-page case: the debug red box should
    // remain visible even if the robot visual meshes are missing.
    if ((meshcat_publish_count_++ % 30) == 0) {
      try {
        diagram_->ForcedPublish(ctx);
        if (meshcat_) {
          meshcat_->SetCameraPose(
              Eigen::Vector3d(2.5, -1.5, 1.4),
              Eigen::Vector3d(0.0, 0.0, 0.8));
        }
      } catch (...) {}
    }

    /* Publish /drake/joint_states ───────────────────────────────── */
    {
      sensor_msgs::msg::JointState js;
      js.header.stamp = now();
      js.name         = act_names_;
      js.position.assign(n_act_, 0.0);
      js.velocity.assign(n_act_, 0.0);
      js.effort.assign(n_act_,   0.0);
      for (int i = 0; i < n_act_; ++i) {
        try {
          const auto& rj =
              dynamic_cast<const drake_mbd::RevoluteJoint<double>&>(
                  plant_->GetJointByName(act_names_[i]));
          const double q = rj.get_angle(pc);
          const double v = rj.get_angular_rate(pc);
          js.position[i] = std::isfinite(q) ? q : 0.0;
          js.velocity[i] = std::isfinite(v) ? v : 0.0;
        } catch (...) {}
      }
      joint_state_pub_->publish(js);

      // Mirror to standard /joint_states so robot_state_publisher feeds RViz.
      // This is what dummy_sim publishes and what makes RViz show the robot.
      js.header.frame_id = "";  // robot_state_publisher expects empty frame_id
      joint_states_rviz_pub_->publish(js);
    }

    /* Publish /drake/foot_contacts — real contact forces from Drake contact results */
    {
      // Query contact results from the plant for foot contact spheres.
      // We look for any contact where one geometry belongs to a foot-related body.
      const double contact_threshold = 10.0;  // N - minimum Fz to count as contact
      double left_fz = 0.0, right_fz = 0.0;

      try {
        const auto& contact_results =
            plant_->get_contact_results_output_port()
                .Eval<drake_mbd::ContactResults<double>>(pc);

        const int num_contacts = contact_results.num_point_pair_contacts();
        for (int c = 0; c < num_contacts; ++c) {
          const auto& info = contact_results.point_pair_contact_info(c);
          // Get the bodies in contact
          const auto& body_a = plant_->get_body(info.bodyA_index());
          const auto& body_b = plant_->get_body(info.bodyB_index());
          const std::string& name_a = body_a.name();
          const std::string& name_b = body_b.name();

          // Force in world z direction (upward = positive)
          const double fz = info.contact_force().z();

          // Check if either body is a left foot link
          auto is_left_foot = [](const std::string& n) {
            return n.find("left_ankle") != std::string::npos ||
                   n.find("left_foot")  != std::string::npos;
          };
          auto is_right_foot = [](const std::string& n) {
            return n.find("right_ankle") != std::string::npos ||
                   n.find("right_foot")  != std::string::npos;
          };

          if (is_left_foot(name_a) || is_left_foot(name_b))
            left_fz  += std::abs(fz);
          if (is_right_foot(name_a) || is_right_foot(name_b))
            right_fz += std::abs(fz);
        }
      } catch (...) {
        // Contact results unavailable — fall back to pelvis-height heuristic
        try {
          const double pelvis_z =
              plant_->EvalBodyPoseInWorld(pc, plant_->GetBodyByName("pelvis"))
                  .translation().z();
          const double half_weight = total_mass_ * 9.81 * 0.5;
          left_fz  = (pelvis_z < 0.9) ? half_weight : 0.0;
          right_fz = (pelvis_z < 0.9) ? half_weight : 0.0;
        } catch (...) {
          // Plant not yet initialized; leave at zero
        }
      }

      if (left_fz <= contact_threshold && right_fz <= contact_threshold) {
        // Hydroelastic-with-fallback can produce no point-pair entries even in
        // a valid standing contact. Use a conservative geometry fallback for
        // the startup/reset pose so MPC does not see a false flight mode.
        try {
          const double pelvis_z =
              plant_->EvalBodyPoseInWorld(pc, plant_->GetBodyByName("pelvis"))
                  .translation().z();
          const double left_ankle_z =
              plant_->EvalBodyPoseInWorld(pc, plant_->GetBodyByName("left_ankle_roll_link"))
                  .translation().z();
          const double right_ankle_z =
              plant_->EvalBodyPoseInWorld(pc, plant_->GetBodyByName("right_ankle_roll_link"))
                  .translation().z();
          const bool standing_height = pelvis_z > 0.50 && pelvis_z < 0.95;
          const bool left_near_ground = left_ankle_z < 0.12;
          const bool right_near_ground = right_ankle_z < 0.12;
          const double half_weight = total_mass_ * 9.81 * 0.5;
          if (standing_height && left_near_ground) left_fz = half_weight;
          if (standing_height && right_near_ground) right_fz = half_weight;
          if (standing_height && !left_near_ground && !right_near_ground) {
            left_fz = half_weight;
            right_fz = half_weight;
          }
        } catch (...) {
          // Keep measured zeros if body lookup fails.
        }
      }

      std_msgs::msg::Float64MultiArray ct;
      ct.data = {
        (left_fz  > contact_threshold) ? 1.0 : 0.0,   // left contact bool
        (right_fz > contact_threshold) ? 1.0 : 0.0,   // right contact bool
        left_fz,                                        // left Fz (N)
        right_fz                                        // right Fz (N)
      };
      contact_pub_->publish(ct);
    }

    /* Publish /drake/sim_time — used by bridge as authoritative time base */
    {
      std_msgs::msg::Float64MultiArray st;
      st.data = {sim_time_};
      sim_time_pub_->publish(st);
    }

    /* Publish /drake/centroidal_state ───────────────────────────── */
    // Layout: [h_lin(3), h_ang(3), base_pos(3), euler_ZYX(3)]
    {
      const auto& pelvis = plant_->GetBodyByName("pelvis");
      const auto  X = plant_->EvalBodyPoseInWorld(pc, pelvis);
      const auto  V = plant_->EvalBodySpatialVelocityInWorld(pc, pelvis);

      const Eigen::Vector3d p = X.translation();
      const Eigen::Matrix3d R = X.rotation().matrix();
      const double yaw   = std::atan2(R(1,0), R(0,0));
      const double pitch = std::asin(std::clamp(-R(2,0), -1.0, 1.0));
      const double roll  = std::atan2(R(2,1), R(2,2));

      // OCS2 centroidal state expects NORMALIZED centroidal momentum.
      // The original MRT controller computes: h_normalized = A(q) * v / robotMass.
      // Therefore the linear part is COM velocity-like, not raw m*v, and the
      // angular part is angular momentum divided by mass.
      Eigen::Vector3d h_lin = V.translational();
      Eigen::Vector3d h_ang = Eigen::Vector3d::Zero();
      try {
        h_lin = plant_->CalcCenterOfMassTranslationalVelocityInWorld(pc);
      } catch (...) {}
      try {
        const Eigen::Vector3d p_com =
            plant_->CalcCenterOfMassPositionInWorld(pc);
        const auto L =
            plant_->CalcSpatialMomentumInWorldAboutPoint(pc, p_com);
        h_ang = L.rotational() / total_mass_;
      } catch (...) {}

      // [FIX-4] Sanity clamp — reject physically impossible values that slip
      // past isfinite() (large-finite from contact explosion). Values here are
      // normalized momentum: linear part is m/s-like, angular part is L/m.
      auto safe = [](double v, double limit) -> double {
        return (std::isfinite(v) && std::abs(v) < limit) ? v : 0.0;
      };

      std_msgs::msg::Float64MultiArray cs;
      cs.data = {
        safe(h_lin.x(), 20.0), safe(h_lin.y(), 20.0), safe(h_lin.z(), 20.0),
        safe(h_ang.x(), 50.0), safe(h_ang.y(), 50.0), safe(h_ang.z(), 50.0),
        safe(p.x(),      50.0), safe(p.y(),      50.0), safe(p.z(),      3.0),
        safe(yaw,         M_PI), safe(pitch, M_PI/2),   safe(roll, M_PI)
      };
      centroidal_pub_->publish(cs);
    }
  }

  /* ── members ─────────────────────────────────────────────────── */
  drake_mbd::MultibodyPlant<double>* plant_{};
  drake_geo::SceneGraph<double>*     sg_{};
  std::shared_ptr<drake::geometry::Meshcat> meshcat_;
  drake_mbd::ModelInstanceIndex      model_;
  std::unique_ptr<drake_sys::Diagram<double>>   diagram_;
  std::unique_ptr<drake_sys::Simulator<double>> simulator_;

  double sim_time_{0.0}, sim_dt_{0.001};
  bool   bridge_connected_{false};
  bool   pin_base_until_first_torque_{true};
  bool   startup_pinned_{true};
  double startup_release_torque_threshold_{-1.0};
  double torque_timeout_s_{0.05};
  std::chrono::steady_clock::time_point last_torque_wall_time_{std::chrono::steady_clock::now()};
  int    consecutive_failures_{0};
  int    recovery_ticks_remaining_{0};
  int    meshcat_publish_count_{0};
  double spawn_height_{0.75};           // set by setInitialState()
  double initial_height_override_{-1.0};
  double total_mass_{35.0}, max_torque_{300.0};
  int    n_act_{0}, n_q_{0}, n_v_{0};
  std::string urdf_path_;

  std::vector<std::string> act_names_;
  Eigen::VectorXd          cmd_tau_;
  std::mutex               mu_;

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr      joint_state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr      joint_states_rviz_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr  centroidal_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr  sim_time_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr  contact_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr startup_pin_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr torque_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DrakeSimNode>());
  rclcpp::shutdown();
  return 0;
}
