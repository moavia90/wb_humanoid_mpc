/******************************************************************************
drake_mrt_sim_node.cpp

Drake + OCS2 MRT runtime for Unitree G1 centroidal MPC.

This version intentionally copies the important stabilizing behaviors from the
repo's MuJoCo simulation interface:

  [1] Auto reset on fall, including zero torque and cooldown.
  [2] Passive joint damping, equivalent to MuJoCo dof_damping = 10.
  [3] Small physics timestep support: sim_dt = 0.0005.
  [4] MuJoCo-equivalent contact flags: RobotState always sees both feet true.
  [5] Direct MuJoCo-style torque application:
        tau = JointAction::getTotalFeedbackTorque(q, qd)
      with no PD/MRT blending.
  [6] Manual compensation for Drake-ignored MJCF features:
        - explicit torque clamps replacing ignored actuatorfrcrange
        - explicit passive damping replacing/augmenting missing dof damping
        - explicit ground contact material/friction properties
        - debug logs for ignored/approximate contact behavior

Important:
  - The real Drake contact forces are still logged separately.
  - The controller contact flags can be forced true to match MuJoCo behavior.
  - startup_pin_duration_s is kept as a Drake warm-start safety mechanism.

******************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <Eigen/Dense>

#include <drake/geometry/meshcat.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/proximity_properties.h>
#include <drake/geometry/rgba.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/shape_specification.h>
#include <drake/math/rigid_transform.h>
#include <drake/math/rotation_matrix.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/contact_results.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/revolute_joint.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/context.h>
#include <drake/systems/framework/diagram_builder.h>

#include <ocs2_sqp/SqpMpc.h>

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h>
#include <humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h>
#include <humanoid_common_mpc_ros2/ros_comm/Ros2ProceduralMpcMotionManager.h>
#include <humanoid_common_mpc_ros2/visualization/HumanoidVisualizer.h>

#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>

namespace drake_mbd = drake::multibody;
namespace drake_geo = drake::geometry;
namespace drake_math = drake::math;
namespace drake_sys = drake::systems;

using namespace ocs2;
using namespace ocs2::humanoid;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

/* -------------------------------------------------------------------------- */
/* Runtime config                                                              */
/* -------------------------------------------------------------------------- */

struct DrakeMrtConfig {
  std::string drake_model_path;

  // [3] MuJoCo-like small physics step.
  double sim_dt{0.0005};

  // Controller loop rate. The node will run multiple Drake substeps per
  // controller tick if control_period > sim_dt.
  double control_rate_hz{250.0};

  // Global hard clamp. Joint-specific limits below are also applied.
  double max_torque{180.0};

  // -1.0 means compute from foot pose. Positive value overrides.
  double initial_height_override{-1.0};

  // Drake-only warm start. Keep 2.0 at first; reduce later after stable.
  double startup_pin_duration_s{2.0};

  // [1] Reset/cooldown behavior, MuJoCo-style.
  bool reset_on_fall{true};
  double reset_cooldown_s{1.0};
  double fall_z_threshold{0.25};
  double fall_angle_threshold_rad{0.90};

  // [2] MuJoCo dof_damping equivalent.
  double passive_joint_damping{10.0};

  // [4] MuJoCo-equivalent contact flags.
  bool force_mujoco_contact_flags{true};

  // [5] Direct torque mode. Keep true to match MuJoCo.
  bool direct_mujoco_torque_mode{true};

  // [6] Manual Drake contact/material compensation.
  double ground_static_friction{0.9};
  double ground_dynamic_friction{0.7};
  double ground_dissipation{2.0};
  double ground_point_stiffness{5e4};
  double ground_hydro_modulus{5e4};
  double ground_hydro_margin{0.05};

  std::string meshcat_host{"0.0.0.0"};

  // Diagnostics.
  double debug_publish_rate_hz{50.0};
  double debug_log_period_s{0.25};
  int debug_top_joints{8};
  bool debug_verbose{true};
};

/* -------------------------------------------------------------------------- */
/* Joint-specific defaults                                                     */
/* -------------------------------------------------------------------------- */

double nominalQ(const std::string& joint_name) {
  if (joint_name.find("hip_pitch") != std::string::npos) return -0.20;
  if (joint_name.find("knee") != std::string::npos) return 0.40;
  if (joint_name.find("ankle_pitch") != std::string::npos) return -0.20;
  return 0.0;
}

// These values compensate for Drake ignoring MJCF actuatorfrcrange /
// actuatorfrclimited. They are conservative and can be tuned.
double nominalTorqueLimit(const std::string& joint_name) {
  if (joint_name.find("hip_") != std::string::npos) return 88.0;
  if (joint_name.find("knee") != std::string::npos) return 139.0;
  if (joint_name.find("ankle") != std::string::npos) return 50.0;

  if (joint_name.find("waist_yaw") != std::string::npos) return 88.0;
  if (joint_name.find("waist_roll") != std::string::npos) return 50.0;
  if (joint_name.find("waist_pitch") != std::string::npos) return 50.0;

  if (joint_name.find("shoulder") != std::string::npos) return 25.0;
  if (joint_name.find("elbow") != std::string::npos) return 25.0;

  if (joint_name.find("wrist_roll") != std::string::npos) return 25.0;
  if (joint_name.find("wrist_pitch") != std::string::npos) return 5.0;
  if (joint_name.find("wrist_yaw") != std::string::npos) return 5.0;

  return 50.0;
}

}  // namespace

class DrakeRobotRuntimeAdapter {
 public:
  DrakeRobotRuntimeAdapter(const std::shared_ptr<rclcpp::Node>& node,
                           const robot::model::RobotDescription& robot_description,
                           const robot::model::RobotState& desired_init_state,
                           const DrakeMrtConfig& config)
      : node_(node),
        robot_description_(robot_description),
        robot_state_(robot_description_, 2),
        robot_joint_action_(robot_description_),
        desired_init_state_(desired_init_state),
        config_(config) {
    buildDiagram();
    setupPublishers();
    setInitialStateFromDesiredState();

    simulator_->Initialize();
    diagram_->ForcedPublish(simulator_->get_context());

    updateRobotStateFromDrake();

    RCLCPP_INFO(
        node_->get_logger(),
        "Drake MRT adapter ready | n_act=%d mass=%.2fkg spawn_z=%.3f "
        "sim_dt=%.6f control_rate=%.1fHz direct_mujoco_torque=%d "
        "passive_damping=%.2f force_mujoco_contacts=%d reset_on_fall=%d",
        n_act_,
        total_mass_,
        spawn_height_,
        config_.sim_dt,
        config_.control_rate_hz,
        config_.direct_mujoco_torque_mode ? 1 : 0,
        config_.passive_joint_damping,
        config_.force_mujoco_contact_flags ? 1 : 0,
        config_.reset_on_fall ? 1 : 0);
  }

  const robot::model::RobotState& getRobotState() const { return robot_state_; }

  robot::model::RobotJointAction& getRobotJointAction() {
    return robot_joint_action_;
  }

  double getSimTime() const { return sim_time_; }

  void markControllerReady() {
    controller_ready_ = true;
    pin_start_time_ = sim_time_;

    RCLCPP_INFO(
        node_->get_logger(),
        "Controller ready at sim_t=%.3f. startup_pin_duration_s=%.3f. "
        "Keep 2.0 initially; reduce only after stable.",
        sim_time_,
        config_.startup_pin_duration_s);
  }

  void stepOnce() {
    auto& root_context = simulator_->get_mutable_context();
    auto& plant_context =
        diagram_->GetMutableSubsystemContext(*plant_, &root_context);

    const bool in_reset_cooldown = reset_cooldown_steps_remaining_ > 0;
    const bool pin_active = startup_pinned_ || in_reset_cooldown;

    if (pin_active) {
      // Drake-only warm start/reset safety. MuJoCo does not pin, but this
      // prevents collapse while MPC/MRT is not ready.
      applyStartupPin(&plant_context);
    }

    if (in_reset_cooldown) {
      --reset_cooldown_steps_remaining_;
      zeroActuation(&plant_context);
    } else if (startup_pinned_) {
      // During pin, keep torques zero. The state is being explicitly held.
      zeroActuation(&plant_context);
    } else {
      // [5] MuJoCo-style: apply controller's JointAction torque directly.
      applyRobotJointActionToDrake(&plant_context);
    }

    sim_time_ += config_.sim_dt;

    try {
      simulator_->AdvanceTo(sim_time_);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(
          node_->get_logger(),
          "Drake AdvanceTo failed at t=%.6f: %s. Resetting state.",
          sim_time_,
          e.what());
      resetDrakeState("AdvanceTo exception");
      return;
    }

    maybeReleaseStartupPin();

    updateRobotStateFromDrake();
    publishDebugTopics();
    logDiagnostics();

    if (config_.reset_on_fall && isFallen()) {
      resetDrakeState("fall_detected");
    }
  }

 private:
  struct ContactDebugInfo {
    bool real_left{false};
    bool real_right{false};

    bool controller_left{true};
    bool controller_right{true};

    int point_pair_contacts{0};
    int hydroelastic_contacts{0};

    double left_fz{0.0};
    double right_fz{0.0};

    double left_ankle_z{std::numeric_limits<double>::quiet_NaN()};
    double right_ankle_z{std::numeric_limits<double>::quiet_NaN()};

    std::string controller_source{"mujoco_forced_true"};
  };

  /* ------------------------------------------------------------------------ */
  /* Diagram/model setup                                                       */
  /* ------------------------------------------------------------------------ */

  void buildDiagram() {
    drake_sys::DiagramBuilder<double> builder;

    auto pair =
        drake_mbd::AddMultibodyPlantSceneGraph(&builder, config_.sim_dt);
    plant_ = &pair.plant;
    scene_graph_ = &pair.scene_graph;

    // Manual compensation for Drake/MJCF differences:
    // Drake ignores some MuJoCo contact fields such as torsional and rolling
    // friction. We set explicit contact material here, but this is still only
    // an approximation, not exact MuJoCo physics.
    plant_->set_contact_model(drake_mbd::ContactModel::kHydroelasticWithFallback);

    drake_mbd::Parser parser(plant_);
    parser.package_map().PopulateFromEnvironment("AMENT_PREFIX_PATH");

    try {
      const std::string share =
          ament_index_cpp::get_package_share_directory("g1_description");
      if (!parser.package_map().Contains("g1_description")) {
        parser.package_map().Add("g1_description", share);
      }
      RCLCPP_INFO(node_->get_logger(), "g1_description package path: %s",
                  share.c_str());
    } catch (const std::exception& e) {
      RCLCPP_WARN(node_->get_logger(),
                  "Could not add g1_description package path: %s", e.what());
    }

    RCLCPP_INFO(node_->get_logger(), "Loading Drake model: %s",
                config_.drake_model_path.c_str());

    model_instance_ = parser.AddModels(config_.drake_model_path).at(0);

    addGround();

    plant_->Finalize();

    n_act_ = plant_->num_actuators();
    if (n_act_ <= 0) {
      throw std::runtime_error("Drake plant has zero actuators.");
    }

    actuation_cmd_ = Eigen::VectorXd::Zero(n_act_);
    raw_controller_tau_ = Eigen::VectorXd::Zero(n_act_);
    damping_tau_ = Eigen::VectorXd::Zero(n_act_);
    unclamped_tau_ = Eigen::VectorXd::Zero(n_act_);

    actuator_names_.reserve(static_cast<std::size_t>(n_act_));
    for (drake_mbd::JointActuatorIndex i(0); i < n_act_; ++i) {
      actuator_names_.push_back(plant_->get_joint_actuator(i).joint().name());
    }

    RCLCPP_INFO(node_->get_logger(), "Drake plant actuators:");
    for (int i = 0; i < n_act_; ++i) {
      RCLCPP_INFO(node_->get_logger(), "  act[%02d] = %s",
                  i, actuator_names_[static_cast<std::size_t>(i)].c_str());
    }

    setupMeshcat(builder);

    diagram_ = builder.Build();
    simulator_ = std::make_unique<drake_sys::Simulator<double>>(*diagram_);
    simulator_->set_target_realtime_rate(0.0);

    const auto& plant_context =
        diagram_->GetSubsystemContext(*plant_, simulator_->get_context());
    total_mass_ = plant_->CalcTotalMass(plant_context);

    RCLCPP_INFO(node_->get_logger(), "Drake Meshcat URL: %s",
                meshcat_->web_url().c_str());
  }

  void addGround() {
    const double ground_size = 20.0;
    const double ground_z = -0.1;

    drake_geo::ProximityProperties props;

    // [6] Approximation for ignored MuJoCo contact settings.
    drake_geo::AddCompliantHydroelasticProperties(
        config_.ground_hydro_margin, config_.ground_hydro_modulus, &props);

    drake_geo::AddContactMaterial(
        config_.ground_dissipation,
        config_.ground_point_stiffness,
        drake_mbd::CoulombFriction<double>(
            config_.ground_static_friction, config_.ground_dynamic_friction),
        &props);

    plant_->RegisterCollisionGeometry(
        plant_->world_body(),
        drake_math::RigidTransformd(Eigen::Vector3d(0.0, 0.0, ground_z)),
        drake_geo::Box(ground_size, ground_size, 0.2),
        "ground_collision",
        props);

    plant_->RegisterVisualGeometry(
        plant_->world_body(),
        drake_math::RigidTransformd(Eigen::Vector3d(0.0, 0.0, ground_z - 0.1)),
        drake_geo::Box(ground_size, ground_size, 0.2),
        "ground_visual",
        Eigen::Vector4d(0.50, 0.50, 0.50, 1.0));

    RCLCPP_INFO(
        node_->get_logger(),
        "Ground contact material | mu_static=%.2f mu_dynamic=%.2f "
        "dissipation=%.2f point_stiffness=%.1e hydro_modulus=%.1e",
        config_.ground_static_friction,
        config_.ground_dynamic_friction,
        config_.ground_dissipation,
        config_.ground_point_stiffness,
        config_.ground_hydro_modulus);
  }

  void setupMeshcat(drake_sys::DiagramBuilder<double>& builder) {
    drake::geometry::MeshcatParams meshcat_params;
    meshcat_params.host = config_.meshcat_host;

    meshcat_ = std::make_shared<drake::geometry::Meshcat>(meshcat_params);

    meshcat_->SetObject("/debug/red_box",
                        drake_geo::Box(0.2, 0.2, 0.2),
                        drake_geo::Rgba(1.0, 0.0, 0.0, 1.0));
    meshcat_->SetTransform(
        "/debug/red_box",
        drake_math::RigidTransformd(Eigen::Vector3d(0.0, 0.0, 1.0)));

    meshcat_->SetCameraPose(Eigen::Vector3d(2.5, -1.5, 1.4),
                            Eigen::Vector3d(0.0, 0.0, 0.8));

    drake_geo::MeshcatVisualizerParams viz_params;
    viz_params.publish_period = 1.0 / 60.0;

    drake_geo::MeshcatVisualizer<double>::AddToBuilder(
        &builder, *scene_graph_, meshcat_, viz_params);
  }

  void setupPublishers() {
    joint_state_pub_ =
        node_->create_publisher<sensor_msgs::msg::JointState>(
            "/drake/joint_states", 10);
    joint_states_rviz_pub_ =
        node_->create_publisher<sensor_msgs::msg::JointState>(
            "/joint_states", 10);

    centroidal_pub_ =
        node_->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/drake/centroidal_state", 10);
    sim_time_pub_ =
        node_->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/drake/sim_time", 10);
    contact_pub_ =
        node_->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/drake/foot_contacts", 10);
  }

  /* ------------------------------------------------------------------------ */
  /* Initial state / reset                                                     */
  /* ------------------------------------------------------------------------ */

  void setInitialStateFromDesiredState() {
    auto& root_context = simulator_->get_mutable_context();
    auto& plant_context =
        diagram_->GetMutableSubsystemContext(*plant_, &root_context);

    applyDesiredStateToPlant(&plant_context, true);

    RCLCPP_INFO(
        node_->get_logger(),
        "Initial state set | spawn_z=%.4f initial_height_override=%.4f",
        spawn_height_,
        config_.initial_height_override);
  }

  void applyDesiredStateToPlant(drake_sys::Context<double>* plant_context,
                                bool recompute_spawn_height) {
    const auto& pelvis = plant_->GetBodyByName("pelvis");

    drake_mbd::SpatialVelocity<double> zero_twist;
    zero_twist.SetZero();

    double target_z = spawn_height_;

    if (recompute_spawn_height) {
      const double reference_height = 1.0;

      plant_->SetFreeBodyPose(
          plant_context,
          pelvis,
          drake_math::RigidTransformd(
              drake_math::RotationMatrixd::Identity(),
              Eigen::Vector3d(0.0, 0.0, reference_height)));

      plant_->SetFreeBodySpatialVelocity(plant_context, pelvis, zero_twist);

      plant_->get_actuation_input_port(model_instance_)
          .FixValue(plant_context, Eigen::VectorXd::Zero(n_act_));

      for (const auto& joint_name : actuator_names_) {
        setJointToInitialValue(plant_context, joint_name);
      }

      double lowest_foot_z = reference_height;
      for (const auto& ankle_name :
           {std::string("left_ankle_roll_link"),
            std::string("right_ankle_roll_link")}) {
        try {
          const auto& ankle_body = plant_->GetBodyByName(ankle_name);
          const auto ankle_pose =
              plant_->EvalBodyPoseInWorld(*plant_context, ankle_body);
          lowest_foot_z =
              std::min(lowest_foot_z, ankle_pose.translation().z() - 0.03);
        } catch (...) {
        }
      }

      constexpr double kFootClearance = 0.005;
      const double foot_offset = reference_height - lowest_foot_z;
      target_z = foot_offset + kFootClearance;

      if (config_.initial_height_override > 0.0) {
        RCLCPP_WARN(
            node_->get_logger(),
            "Manual initial_height override %.4f m; auto was %.4f m.",
            config_.initial_height_override,
            target_z);
        target_z = config_.initial_height_override;
      }

      spawn_height_ = target_z;
      desired_root_position_ = Eigen::Vector3d(0.0, 0.0, spawn_height_);
    }

    plant_->SetFreeBodyPose(
        plant_context,
        pelvis,
        drake_math::RigidTransformd(
            drake_math::RotationMatrixd::Identity(),
            Eigen::Vector3d(0.0, 0.0, target_z)));

    plant_->SetFreeBodySpatialVelocity(plant_context, pelvis, zero_twist);

    for (const auto& joint_name : actuator_names_) {
      setJointToInitialValue(plant_context, joint_name);
    }
  }

  void setJointToInitialValue(drake_sys::Context<double>* plant_context,
                              const std::string& joint_name) {
    try {
      const auto& joint =
          dynamic_cast<const drake_mbd::RevoluteJoint<double>&>(
              plant_->GetJointByName(joint_name));

      double q0 = nominalQ(joint_name);

      if (robot_description_.containsJoint(joint_name)) {
        q0 = desired_init_state_.getJointPosition(
            robot_description_.getJointIndex(joint_name));
      }

      joint.set_angle(plant_context, q0);
      joint.set_angular_rate(plant_context, 0.0);
    } catch (...) {
    }
  }

  void applyStartupPin(drake_sys::Context<double>* plant_context) {
    // Drake-only safety. It overwrites floating-base and joint state so the
    // controller can initialize without the robot falling.
    applyDesiredStateToPlant(plant_context, false);
  }

  void maybeReleaseStartupPin() {
    if (!startup_pinned_) {
      return;
    }

    if (!controller_ready_) {
      return;
    }

    if (reset_cooldown_steps_remaining_ > 0) {
      return;
    }

    if ((sim_time_ - pin_start_time_) >= config_.startup_pin_duration_s) {
      startup_pinned_ = false;
      RCLCPP_WARN(
          node_->get_logger(),
          "Startup pin released at sim_t=%.3f. Direct MuJoCo-style torque is now active.",
          sim_time_);
    }
  }

  void resetDrakeState(const std::string& reason) {
    auto& root_context = simulator_->get_mutable_context();
    auto& plant_context =
        diagram_->GetMutableSubsystemContext(*plant_, &root_context);

    ++reset_count_;

    RCLCPP_ERROR(
        node_->get_logger(),
        "[drake_reset] reason=%s count=%d sim_t=%.3f. "
        "Resetting base, joints, velocities, torques, and entering cooldown %.2fs.",
        reason.c_str(),
        reset_count_,
        sim_time_,
        config_.reset_cooldown_s);

    applyDesiredStateToPlant(&plant_context, false);
    zeroActuation(&plant_context);

    startup_pinned_ = true;
    pin_start_time_ = sim_time_;

    reset_cooldown_steps_remaining_ =
        std::max(1, static_cast<int>(
                        std::round(config_.reset_cooldown_s / config_.sim_dt)));

    try {
      simulator_->Initialize();
    } catch (const std::exception& e) {
      RCLCPP_ERROR(node_->get_logger(),
                   "simulator_->Initialize() after reset failed: %s", e.what());
    }

    updateRobotStateFromDrake();
  }

  bool isFallen() const {
    const auto& context = simulator_->get_context();
    const auto& plant_context = diagram_->GetSubsystemContext(*plant_, context);

    try {
      const auto& pelvis = plant_->GetBodyByName("pelvis");
      const auto X = plant_->EvalBodyPoseInWorld(plant_context, pelvis);
      const auto R = X.rotation().matrix();

      const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
      const double roll = std::atan2(R(2, 1), R(2, 2));

      return X.translation().z() < config_.fall_z_threshold ||
             std::abs(roll) > config_.fall_angle_threshold_rad ||
             std::abs(pitch) > config_.fall_angle_threshold_rad;
    } catch (...) {
      return false;
    }
  }

  /* ------------------------------------------------------------------------ */
  /* State mapping                                                             */
  /* ------------------------------------------------------------------------ */

  void updateRobotStateFromDrake() {
    const auto& root_context = simulator_->get_context();
    const auto& plant_context =
        diagram_->GetSubsystemContext(*plant_, root_context);
    const auto& pelvis = plant_->GetBodyByName("pelvis");

    const auto X_WB = plant_->EvalBodyPoseInWorld(plant_context, pelvis);
    const auto V_WB =
        plant_->EvalBodySpatialVelocityInWorld(plant_context, pelvis);

    robot::quaternion_t quat_l_to_w(X_WB.rotation().matrix());

    robot_state_.setRootRotationLocalToWorldFrame(quat_l_to_w);
    robot_state_.setRootPositionInWorldFrame(X_WB.translation());

    robot_state_.setRootLinearVelocityInLocalFrame(
        quat_l_to_w.inverse() * V_WB.translational());

    robot_state_.setRootAngularVelocityInLocalFrame(
        quat_l_to_w.inverse() * V_WB.rotational());

    for (const auto& joint_name : robot_description_.getJointNames()) {
      const auto joint_idx = robot_description_.getJointIndex(joint_name);

      try {
        const auto& joint =
            dynamic_cast<const drake_mbd::RevoluteJoint<double>&>(
                plant_->GetJointByName(joint_name));

        robot_state_.setJointPosition(joint_idx, joint.get_angle(plant_context));
        robot_state_.setJointVelocity(joint_idx,
                                      joint.get_angular_rate(plant_context));
      } catch (...) {
        robot_state_.setJointPosition(joint_idx, 0.0);
        robot_state_.setJointVelocity(joint_idx, 0.0);
      }
    }

    // [4] This matches MuJoCo code behavior: contact sensors are ignored and
    // both feet are always true. Real Drake contact forces are still logged.
    if (config_.force_mujoco_contact_flags) {
      robot_state_.setContactFlag(0, true);
      robot_state_.setContactFlag(1, true);
    } else {
      const auto contact = computeContactDebugInfo(plant_context);
      robot_state_.setContactFlag(0, contact.real_left);
      robot_state_.setContactFlag(1, contact.real_right);
    }

    robot_state_.setTime(sim_time_);
  }

  /* ------------------------------------------------------------------------ */
  /* Contact debug                                                             */
  /* ------------------------------------------------------------------------ */

  ContactDebugInfo computeContactDebugInfo(
      const drake_sys::Context<double>& plant_context) const {
    ContactDebugInfo info;

    try {
      const auto& left_ankle = plant_->GetBodyByName("left_ankle_roll_link");
      const auto& right_ankle = plant_->GetBodyByName("right_ankle_roll_link");

      info.left_ankle_z =
          plant_->EvalBodyPoseInWorld(plant_context, left_ankle)
              .translation()
              .z();

      info.right_ankle_z =
          plant_->EvalBodyPoseInWorld(plant_context, right_ankle)
              .translation()
              .z();
    } catch (...) {
    }

    constexpr double kContactThresholdN = 10.0;

    try {
      const auto& contact_results =
          plant_->get_contact_results_output_port()
              .Eval<drake_mbd::ContactResults<double>>(plant_context);

      info.point_pair_contacts = contact_results.num_point_pair_contacts();
      info.hydroelastic_contacts = contact_results.num_hydroelastic_contacts();

      for (int c = 0; c < contact_results.num_point_pair_contacts(); ++c) {
        const auto& contact = contact_results.point_pair_contact_info(c);
        const auto& body_a = plant_->get_body(contact.bodyA_index());
        const auto& body_b = plant_->get_body(contact.bodyB_index());

        const auto is_left = [](const std::string& name) {
          return name.find("left_ankle") != std::string::npos ||
                 name.find("left_foot") != std::string::npos;
        };

        const auto is_right = [](const std::string& name) {
          return name.find("right_ankle") != std::string::npos ||
                 name.find("right_foot") != std::string::npos;
        };

        const double fz = std::abs(contact.contact_force().z());

        if (is_left(body_a.name()) || is_left(body_b.name())) {
          info.left_fz += fz;
        }

        if (is_right(body_a.name()) || is_right(body_b.name())) {
          info.right_fz += fz;
        }
      }
    } catch (...) {
    }

    info.real_left = info.left_fz > kContactThresholdN;
    info.real_right = info.right_fz > kContactThresholdN;

    if (config_.force_mujoco_contact_flags) {
      info.controller_left = true;
      info.controller_right = true;
      info.controller_source = "mujoco_forced_true";
    } else {
      info.controller_left = info.real_left;
      info.controller_right = info.real_right;
      info.controller_source = "drake_contact_force";
    }

    return info;
  }

  /* ------------------------------------------------------------------------ */
  /* Actuation                                                                 */
  /* ------------------------------------------------------------------------ */

  void zeroActuation(drake_sys::Context<double>* plant_context) {
    actuation_cmd_.setZero();
    raw_controller_tau_.setZero();
    damping_tau_.setZero();
    unclamped_tau_.setZero();

    torque_inf_ = 0.0;
    raw_tau_inf_ = 0.0;
    damping_tau_inf_ = 0.0;
    missing_action_count_ = 0;
    nonfinite_action_count_ = 0;
    saturation_count_ = 0;
    saturated_joint_summary_ = "none";

    plant_->get_actuation_input_port(model_instance_)
        .FixValue(plant_context, actuation_cmd_);
  }

  void applyRobotJointActionToDrake(drake_sys::Context<double>* plant_context) {
    raw_controller_tau_.setZero();
    damping_tau_.setZero();
    unclamped_tau_.setZero();
    actuation_cmd_.setZero();

    missing_action_count_ = 0;
    nonfinite_action_count_ = 0;
    saturation_count_ = 0;

    // [5] Direct MuJoCo-style torque:
    //   ctrl[i] = jointAction.getTotalFeedbackTorque(q, qd)
    // This intentionally removes the earlier PD/MRT blend.
    for (int ai = 0; ai < n_act_; ++ai) {
      const std::string& joint_name =
          actuator_names_[static_cast<std::size_t>(ai)];

      if (!robot_description_.containsJoint(joint_name)) {
        continue;
      }

      const auto joint_idx = robot_description_.getJointIndex(joint_name);

      double q_meas = 0.0;
      double qd_meas = 0.0;

      try {
        const auto& joint =
            dynamic_cast<const drake_mbd::RevoluteJoint<double>&>(
                plant_->GetJointByName(joint_name));
        q_meas = joint.get_angle(*plant_context);
        qd_meas = joint.get_angular_rate(*plant_context);
      } catch (...) {
      }

      double tau_controller = 0.0;

      if (config_.direct_mujoco_torque_mode) {
        const auto& action_opt = robot_joint_action_.at(joint_idx);

        if (!action_opt.has_value()) {
          ++missing_action_count_;
          tau_controller = 0.0;
        } else {
          tau_controller =
              action_opt.value().getTotalFeedbackTorque(q_meas, qd_meas);
        }

        if (!std::isfinite(tau_controller)) {
          ++nonfinite_action_count_;
          tau_controller = 0.0;
        }
      }

      // [2] Passive joint damping equivalent to MuJoCo dof_damping = 10.
      const double tau_damping = -config_.passive_joint_damping * qd_meas;

      raw_controller_tau_[ai] = tau_controller;
      damping_tau_[ai] = tau_damping;
      unclamped_tau_[ai] = tau_controller + tau_damping;
    }

    Eigen::VectorXd clamped = unclamped_tau_;

    for (int ai = 0; ai < n_act_; ++ai) {
      const std::string& joint_name =
          actuator_names_[static_cast<std::size_t>(ai)];

      const double limit =
          std::min(config_.max_torque, nominalTorqueLimit(joint_name));

      if (!std::isfinite(clamped[ai])) {
        clamped[ai] = 0.0;
      }

      const double before = clamped[ai];
      clamped[ai] = std::clamp(clamped[ai], -limit, limit);

      if (std::abs(before - clamped[ai]) > 1e-9) {
        ++saturation_count_;
      }
    }

    actuation_cmd_ = clamped;

    raw_tau_inf_ =
        raw_controller_tau_.size() > 0 ? raw_controller_tau_.cwiseAbs().maxCoeff() : 0.0;
    damping_tau_inf_ =
        damping_tau_.size() > 0 ? damping_tau_.cwiseAbs().maxCoeff() : 0.0;
    torque_inf_ =
        actuation_cmd_.size() > 0 ? actuation_cmd_.cwiseAbs().maxCoeff() : 0.0;

    saturated_joint_summary_ =
        saturatedJointsString(unclamped_tau_, actuation_cmd_,
                              config_.debug_top_joints);

    plant_->get_actuation_input_port(model_instance_)
        .FixValue(plant_context, actuation_cmd_);
  }

  /* ------------------------------------------------------------------------ */
  /* Debug publishing/logging                                                  */
  /* ------------------------------------------------------------------------ */

  void publishDebugTopics() {
    const int every_n = std::max(
        1,
        static_cast<int>(
            std::round(1.0 / (config_.sim_dt * config_.debug_publish_rate_hz))));

    if ((publish_count_++ % every_n) != 0) {
      return;
    }

    const auto& root_context = simulator_->get_context();
    const auto& plant_context =
        diagram_->GetSubsystemContext(*plant_, root_context);

    publishJointStates(plant_context);
    publishContacts(plant_context);
    publishCentroidalState(plant_context);
    publishSimTime();

    try {
      diagram_->ForcedPublish(root_context);
    } catch (...) {
    }
  }

  void publishJointStates(const drake_sys::Context<double>& plant_context) {
    sensor_msgs::msg::JointState js;
    js.header.stamp = node_->now();
    js.name = actuator_names_;
    js.position.assign(static_cast<std::size_t>(n_act_), 0.0);
    js.velocity.assign(static_cast<std::size_t>(n_act_), 0.0);
    js.effort.assign(static_cast<std::size_t>(n_act_), 0.0);

    for (int ai = 0; ai < n_act_; ++ai) {
      try {
        const auto& joint =
            dynamic_cast<const drake_mbd::RevoluteJoint<double>&>(
                plant_->GetJointByName(
                    actuator_names_[static_cast<std::size_t>(ai)]));
        js.position[static_cast<std::size_t>(ai)] =
            joint.get_angle(plant_context);
        js.velocity[static_cast<std::size_t>(ai)] =
            joint.get_angular_rate(plant_context);
        js.effort[static_cast<std::size_t>(ai)] = actuation_cmd_[ai];
      } catch (...) {
      }
    }

    joint_state_pub_->publish(js);
    joint_states_rviz_pub_->publish(js);
  }

  void publishContacts(const drake_sys::Context<double>& plant_context) {
    const auto contact = computeContactDebugInfo(plant_context);

    // Layout:
    // [controller_left, controller_right, real_left_fz, real_right_fz,
    //  real_left_bool, real_right_bool, point_pair_count, hydro_count]
    std_msgs::msg::Float64MultiArray ct;
    ct.data = {
        contact.controller_left ? 1.0 : 0.0,
        contact.controller_right ? 1.0 : 0.0,
        contact.left_fz,
        contact.right_fz,
        contact.real_left ? 1.0 : 0.0,
        contact.real_right ? 1.0 : 0.0,
        static_cast<double>(contact.point_pair_contacts),
        static_cast<double>(contact.hydroelastic_contacts),
    };

    contact_pub_->publish(ct);
  }

  void publishCentroidalState(const drake_sys::Context<double>& plant_context) {
    const auto& pelvis = plant_->GetBodyByName("pelvis");
    const auto X = plant_->EvalBodyPoseInWorld(plant_context, pelvis);
    const auto R = X.rotation().matrix();

    const double yaw = std::atan2(R(1, 0), R(0, 0));
    const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
    const double roll = std::atan2(R(2, 1), R(2, 2));

    Eigen::Vector3d h_lin = Eigen::Vector3d::Zero();
    Eigen::Vector3d h_ang = Eigen::Vector3d::Zero();

    try {
      h_lin =
          total_mass_ *
          plant_->CalcCenterOfMassTranslationalVelocityInWorld(plant_context);

      const Eigen::Vector3d p_com =
          plant_->CalcCenterOfMassPositionInWorld(plant_context);
      const auto L =
          plant_->CalcSpatialMomentumInWorldAboutPoint(plant_context, p_com);
      h_ang = L.rotational();
    } catch (...) {
    }

    auto safe = [](double v, double limit) {
      return (std::isfinite(v) && std::abs(v) < limit) ? v : 0.0;
    };

    // Layout: [h_lin(3), h_ang(3), base_pos(3), yaw,pitch,roll]
    std_msgs::msg::Float64MultiArray cs;
    cs.data = {
        safe(h_lin.x(), 2000.0),
        safe(h_lin.y(), 2000.0),
        safe(h_lin.z(), 2000.0),
        safe(h_ang.x(), 2000.0),
        safe(h_ang.y(), 2000.0),
        safe(h_ang.z(), 2000.0),
        safe(X.translation().x(), 50.0),
        safe(X.translation().y(), 50.0),
        safe(X.translation().z(), 5.0),
        safe(yaw, kPi),
        safe(pitch, kPi / 2.0),
        safe(roll, kPi),
    };

    centroidal_pub_->publish(cs);
  }

  void publishSimTime() {
    std_msgs::msg::Float64MultiArray st;
    st.data = {sim_time_};
    sim_time_pub_->publish(st);
  }

  std::string topAbsJointsString(const Eigen::VectorXd& v, int top_n) const {
    std::vector<std::pair<double, int>> vals;
    vals.reserve(static_cast<std::size_t>(v.size()));

    for (int i = 0; i < v.size(); ++i) {
      vals.emplace_back(std::abs(v[i]), i);
    }

    std::sort(vals.begin(), vals.end(),
              [](const auto& a, const auto& b) {
                return a.first > b.first;
              });

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);

    const int n = std::min(top_n, static_cast<int>(vals.size()));
    for (int k = 0; k < n; ++k) {
      const int idx = vals[static_cast<std::size_t>(k)].second;
      ss << actuator_names_[static_cast<std::size_t>(idx)]
         << "=" << v[idx];

      if (k + 1 < n) {
        ss << ", ";
      }
    }

    return ss.str();
  }

  std::string saturatedJointsString(const Eigen::VectorXd& unclamped,
                                    const Eigen::VectorXd& clamped,
                                    int top_n) const {
    std::vector<std::pair<double, int>> vals;

    for (int i = 0; i < unclamped.size(); ++i) {
      const double diff = std::abs(unclamped[i] - clamped[i]);
      if (diff > 1e-9) {
        vals.emplace_back(diff, i);
      }
    }

    std::sort(vals.begin(), vals.end(),
              [](const auto& a, const auto& b) {
                return a.first > b.first;
              });

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);

    const int n = std::min(top_n, static_cast<int>(vals.size()));
    for (int k = 0; k < n; ++k) {
      const int idx = vals[static_cast<std::size_t>(k)].second;

      ss << actuator_names_[static_cast<std::size_t>(idx)]
         << ": raw=" << unclamped[idx]
         << " clamp=" << clamped[idx];

      if (k + 1 < n) {
        ss << ", ";
      }
    }

    if (vals.empty()) {
      ss << "none";
    }

    return ss.str();
  }

  void logDiagnostics() {
    if (!config_.debug_verbose) {
      return;
    }

    const int every_n = std::max(
        1, static_cast<int>(
               std::round(config_.debug_log_period_s / config_.sim_dt)));

    if ((diag_count_++ % every_n) != 0) {
      return;
    }

    const auto& root_context = simulator_->get_context();
    const auto& plant_context =
        diagram_->GetSubsystemContext(*plant_, root_context);

    const auto& pelvis = plant_->GetBodyByName("pelvis");
    const auto X = plant_->EvalBodyPoseInWorld(plant_context, pelvis);
    const auto V =
        plant_->EvalBodySpatialVelocityInWorld(plant_context, pelvis);

    const auto R = X.rotation().matrix();

    const double yaw = std::atan2(R(1, 0), R(0, 0));
    const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
    const double roll = std::atan2(R(2, 1), R(2, 2));

    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    Eigen::Vector3d com_v = Eigen::Vector3d::Zero();

    try {
      com = plant_->CalcCenterOfMassPositionInWorld(plant_context);
      com_v =
          plant_->CalcCenterOfMassTranslationalVelocityInWorld(plant_context);
    } catch (...) {
    }

    const auto contact = computeContactDebugInfo(plant_context);
    const bool fallen_now = isFallen();

    const char* phase =
        reset_cooldown_steps_remaining_ > 0
            ? "RESET_COOLDOWN"
            : (startup_pinned_ ? "STARTUP_PIN" : "DIRECT_MUJOCO_TORQUE");

    RCLCPP_INFO(
        node_->get_logger(),
        "\n[drake_mrt_diag]\n"
        "  t=%.3f phase=%s fallen=%d reset_count=%d pin=%d cooldown_steps=%d\n"
        "  base: pos=[%.3f %.3f %.3f] rpy=[%.3f %.3f %.3f] "
        "linvel_norm=%.3f angvel_norm=%.3f\n"
        "  com : pos=[%.3f %.3f %.3f] vel=[%.3f %.3f %.3f]\n"
        "  contacts: controller=[%d %d] source=%s real=[%d %d] "
        "fz=[%.1f %.1f] point_contacts=%d hydro_contacts=%d ankle_z=[%.3f %.3f]\n"
        "  torque: final_inf=%.1f raw_controller_inf=%.1f damping_inf=%.1f "
        "sat=%d missing_actions=%d nonfinite_actions=%d passive_damping=%.2f\n"
        "  top_final_tau     : %s\n"
        "  top_raw_controller: %s\n"
        "  top_damping_tau   : %s\n"
        "  saturated         : %s",
        sim_time_,
        phase,
        fallen_now ? 1 : 0,
        reset_count_,
        startup_pinned_ ? 1 : 0,
        reset_cooldown_steps_remaining_,

        X.translation().x(),
        X.translation().y(),
        X.translation().z(),
        roll,
        pitch,
        yaw,
        V.translational().norm(),
        V.rotational().norm(),

        com.x(),
        com.y(),
        com.z(),
        com_v.x(),
        com_v.y(),
        com_v.z(),

        contact.controller_left ? 1 : 0,
        contact.controller_right ? 1 : 0,
        contact.controller_source.c_str(),
        contact.real_left ? 1 : 0,
        contact.real_right ? 1 : 0,
        contact.left_fz,
        contact.right_fz,
        contact.point_pair_contacts,
        contact.hydroelastic_contacts,
        contact.left_ankle_z,
        contact.right_ankle_z,

        torque_inf_,
        raw_tau_inf_,
        damping_tau_inf_,
        saturation_count_,
        missing_action_count_,
        nonfinite_action_count_,
        config_.passive_joint_damping,

        topAbsJointsString(actuation_cmd_, config_.debug_top_joints).c_str(),
        topAbsJointsString(raw_controller_tau_, config_.debug_top_joints).c_str(),
        topAbsJointsString(damping_tau_, config_.debug_top_joints).c_str(),
        saturated_joint_summary_.c_str());
  }

  /* ------------------------------------------------------------------------ */
  /* Members                                                                   */
  /* ------------------------------------------------------------------------ */

  std::shared_ptr<rclcpp::Node> node_;
  const robot::model::RobotDescription& robot_description_;

  robot::model::RobotState robot_state_;
  robot::model::RobotJointAction robot_joint_action_;
  robot::model::RobotState desired_init_state_;

  DrakeMrtConfig config_;

  drake_mbd::MultibodyPlant<double>* plant_{nullptr};
  drake_geo::SceneGraph<double>* scene_graph_{nullptr};
  drake_mbd::ModelInstanceIndex model_instance_;

  std::shared_ptr<drake_geo::Meshcat> meshcat_;
  std::unique_ptr<drake_sys::Diagram<double>> diagram_;
  std::unique_ptr<drake_sys::Simulator<double>> simulator_;

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_rviz_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr centroidal_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr sim_time_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr contact_pub_;

  std::vector<std::string> actuator_names_;

  Eigen::VectorXd actuation_cmd_;
  Eigen::VectorXd raw_controller_tau_;
  Eigen::VectorXd damping_tau_;
  Eigen::VectorXd unclamped_tau_;

  Eigen::Vector3d desired_root_position_{0.0, 0.0, 0.8};

  double total_mass_{35.0};
  double spawn_height_{0.8};
  double sim_time_{0.0};

  bool controller_ready_{false};
  bool startup_pinned_{true};

  double pin_start_time_{0.0};

  int reset_cooldown_steps_remaining_{0};
  int reset_count_{0};

  int n_act_{0};
  int publish_count_{0};
  int diag_count_{0};

  double torque_inf_{0.0};
  double raw_tau_inf_{0.0};
  double damping_tau_inf_{0.0};

  int saturation_count_{0};
  int missing_action_count_{0};
  int nonfinite_action_count_{0};

  std::string saturated_joint_summary_{"none"};
};

/* ----------------------------------------------------------------------------
 * main()
 * -------------------------------------------------------------------------- */

int main(int argc, char** argv) {
  const std::vector<std::string> program_args =
      rclcpp::remove_ros_arguments(argc, argv);

  if (program_args.size() < 7) {
    throw std::runtime_error(
        "Expected args: robot_name task_file reference_file urdf_file "
        "gait_file drake_model_file");
  }

  const std::string robot_name = program_args[1];
  const std::string task_file = program_args[2];
  const std::string reference_file = program_args[3];
  const std::string urdf_file = program_args[4];
  const std::string gait_file = program_args[5];
  const std::string drake_model_file = program_args[6];

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(robot_name + "_drake_mrt_sim");

  node->declare_parameter<double>("sim_dt", 0.0005);
  node->declare_parameter<double>("control_rate_hz", 250.0);
  node->declare_parameter<double>("max_torque", 180.0);
  node->declare_parameter<double>("initial_height", -1.0);

  node->declare_parameter<double>("startup_pin_duration_s", 2.0);

  node->declare_parameter<bool>("reset_on_fall", true);
  node->declare_parameter<double>("reset_cooldown_s", 1.0);
  node->declare_parameter<double>("fall_z_threshold", 0.25);
  node->declare_parameter<double>("fall_angle_threshold_rad", 0.90);

  node->declare_parameter<double>("passive_joint_damping", 10.0);
  node->declare_parameter<bool>("force_mujoco_contact_flags", true);
  node->declare_parameter<bool>("direct_mujoco_torque_mode", true);

  node->declare_parameter<double>("ground_static_friction", 0.9);
  node->declare_parameter<double>("ground_dynamic_friction", 0.7);
  node->declare_parameter<double>("ground_dissipation", 2.0);
  node->declare_parameter<double>("ground_point_stiffness", 5e4);
  node->declare_parameter<double>("ground_hydro_modulus", 5e4);
  node->declare_parameter<double>("ground_hydro_margin", 0.05);

  node->declare_parameter<std::string>("meshcat_host", "0.0.0.0");

  node->declare_parameter<double>("debug_publish_rate_hz", 50.0);
  node->declare_parameter<double>("debug_log_period_s", 0.25);
  node->declare_parameter<int>("debug_top_joints", 8);
  node->declare_parameter<bool>("debug_verbose", true);

  DrakeMrtConfig cfg;
  cfg.drake_model_path = drake_model_file;

  cfg.sim_dt = node->get_parameter("sim_dt").as_double();
  cfg.control_rate_hz = node->get_parameter("control_rate_hz").as_double();
  cfg.max_torque = node->get_parameter("max_torque").as_double();
  cfg.initial_height_override =
      node->get_parameter("initial_height").as_double();

  cfg.startup_pin_duration_s =
      node->get_parameter("startup_pin_duration_s").as_double();

  cfg.reset_on_fall = node->get_parameter("reset_on_fall").as_bool();
  cfg.reset_cooldown_s =
      node->get_parameter("reset_cooldown_s").as_double();
  cfg.fall_z_threshold =
      node->get_parameter("fall_z_threshold").as_double();
  cfg.fall_angle_threshold_rad =
      node->get_parameter("fall_angle_threshold_rad").as_double();

  cfg.passive_joint_damping =
      node->get_parameter("passive_joint_damping").as_double();
  cfg.force_mujoco_contact_flags =
      node->get_parameter("force_mujoco_contact_flags").as_bool();
  cfg.direct_mujoco_torque_mode =
      node->get_parameter("direct_mujoco_torque_mode").as_bool();

  cfg.ground_static_friction =
      node->get_parameter("ground_static_friction").as_double();
  cfg.ground_dynamic_friction =
      node->get_parameter("ground_dynamic_friction").as_double();
  cfg.ground_dissipation =
      node->get_parameter("ground_dissipation").as_double();
  cfg.ground_point_stiffness =
      node->get_parameter("ground_point_stiffness").as_double();
  cfg.ground_hydro_modulus =
      node->get_parameter("ground_hydro_modulus").as_double();
  cfg.ground_hydro_margin =
      node->get_parameter("ground_hydro_margin").as_double();

  cfg.meshcat_host = node->get_parameter("meshcat_host").as_string();

  cfg.debug_publish_rate_hz =
      node->get_parameter("debug_publish_rate_hz").as_double();
  cfg.debug_log_period_s =
      node->get_parameter("debug_log_period_s").as_double();
  cfg.debug_top_joints =
      node->get_parameter("debug_top_joints").as_int();
  cfg.debug_verbose =
      node->get_parameter("debug_verbose").as_bool();

  RCLCPP_INFO(
      node->get_logger(),
      "Args:\n"
      "  robot=%s\n"
      "  task=%s\n"
      "  reference=%s\n"
      "  urdf=%s\n"
      "  gait=%s\n"
      "  drake_model=%s",
      robot_name.c_str(),
      task_file.c_str(),
      reference_file.c_str(),
      urdf_file.c_str(),
      gait_file.c_str(),
      drake_model_file.c_str());

  RCLCPP_INFO(
      node->get_logger(),
      "Runtime config:\n"
      "  sim_dt=%.6f control_rate_hz=%.1f\n"
      "  startup_pin_duration_s=%.3f reset_on_fall=%d reset_cooldown_s=%.3f\n"
      "  passive_joint_damping=%.2f force_mujoco_contact_flags=%d "
      "direct_mujoco_torque_mode=%d\n"
      "  ground mu=[%.2f %.2f] dissipation=%.2f point_stiffness=%.1e "
      "hydro_modulus=%.1e",
      cfg.sim_dt,
      cfg.control_rate_hz,
      cfg.startup_pin_duration_s,
      cfg.reset_on_fall ? 1 : 0,
      cfg.reset_cooldown_s,
      cfg.passive_joint_damping,
      cfg.force_mujoco_contact_flags ? 1 : 0,
      cfg.direct_mujoco_torque_mode ? 1 : 0,
      cfg.ground_static_friction,
      cfg.ground_dynamic_friction,
      cfg.ground_dissipation,
      cfg.ground_point_stiffness,
      cfg.ground_hydro_modulus);

  CentroidalMpcInterface interface(task_file, urdf_file, reference_file);

  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(),
             interface.getOptimalControlProblem(),
             interface.getInitializer());

  auto qos = rclcpp::QoS(1).best_effort();

  auto humanoid_visualizer = std::make_shared<HumanoidVisualizer>(
      task_file,
      interface.getPinocchioInterface(),
      interface.getMpcRobotModel(),
      node);

  CentroidalMpcTargetTrajectoriesCalculator target_calc(
      reference_file,
      interface.getMpcRobotModel(),
      interface.getPinocchioInterface(),
      interface.getCentroidalModelInfo(),
      interface.mpcSettings().timeHorizon_);

  ProceduralMpcMotionManager::VelocityTargetToTargetTrajectories target_func =
      [&target_calc](const vector4_t& velocity_target,
                     scalar_t init_time,
                     scalar_t final_time,
                     const vector_t& init_state) mutable {
        (void)final_time;
        return target_calc.commandedVelocityToTargetTrajectories(
            velocity_target, init_time, init_state);
      };

  auto motion_manager = std::make_shared<Ros2ProceduralMpcMotionManager>(
      gait_file,
      reference_file,
      interface.getSwitchedModelReferenceManagerPtr(),
      interface.getMpcRobotModel(),
      target_func);

  motion_manager->subscribe(node, qos);

  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());
  mpc.getSolverPtr()->addSynchronizedModule(motion_manager);

  robot::model::RobotDescription robot_description(urdf_file);

  robot::model::RobotState init_state(robot_description, 2);
  init_state.setConfigurationToZero();

  const vector_t& init_mpc_state = interface.getInitialState();
  const auto& mpc_model = interface.getMpcRobotModel();

  init_state.setRootPositionInWorldFrame(
      mpc_model.getBasePosition(init_mpc_state));

  const vector_t mpc_joint_angles =
      mpc_model.getJointAngles(init_mpc_state);

  const auto mpc_joint_indices =
      robot_description.getJointIndices(
          interface.modelSettings().mpcModelJointNames);

  for (size_t i = 0; i < mpc_joint_indices.size(); ++i) {
    init_state.setJointPosition(
        mpc_joint_indices[i],
        mpc_joint_angles[static_cast<Eigen::Index>(i)]);
  }

  DrakeRobotRuntimeAdapter adapter(
      node, robot_description, init_state, cfg);

  CentroidalMpcMrtJointController controller(
      robot_description,
      interface.modelSettings(),
      interface.getMpcRobotModel(),
      mpc,
      interface.getPinocchioInterface(),
      interface.mpcSettings().mpcDesiredFrequency_,
      humanoid_visualizer);

  controller.startMpcThread(adapter.getRobotState());

  while (rclcpp::ok() && !controller.ready()) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  adapter.markControllerReady();

  RCLCPP_INFO(node->get_logger(),
              "Initial MRT policy received; entering Drake control loop.");

  const double control_period_s = 1.0 / cfg.control_rate_hz;

  const int physics_steps_per_control =
      std::max(1, static_cast<int>(std::round(control_period_s / cfg.sim_dt)));

  const auto control_dt =
      std::chrono::microseconds(
          static_cast<int>(1e6 / cfg.control_rate_hz));

  RCLCPP_INFO(
      node->get_logger(),
      "Loop schedule: control_period=%.6fs sim_dt=%.6fs "
      "physics_steps_per_control=%d simulated_dt_per_control=%.6fs",
      control_period_s,
      cfg.sim_dt,
      physics_steps_per_control,
      physics_steps_per_control * cfg.sim_dt);

  int loop_count = 0;

  while (rclcpp::ok()) {
    const auto loop_start = std::chrono::steady_clock::now();
    const auto next_tick = loop_start + control_dt;

    const auto t0 = std::chrono::steady_clock::now();

    controller.computeJointControlAction(
        adapter.getRobotState().getTime(),
        adapter.getRobotState(),
        adapter.getRobotJointAction());

    const auto t1 = std::chrono::steady_clock::now();

    for (int i = 0; i < physics_steps_per_control; ++i) {
      adapter.stepOnce();
    }

    const auto t2 = std::chrono::steady_clock::now();

    rclcpp::spin_some(node);

    const auto t3 = std::chrono::steady_clock::now();

    const double mrt_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double step_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double spin_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();
    const double total_ms =
        std::chrono::duration<double, std::milli>(t3 - loop_start).count();

    const double budget_ms = 1000.0 / cfg.control_rate_hz;
    const double overrun_ms = total_ms - budget_ms;

    const double sim_advanced_s = physics_steps_per_control * cfg.sim_dt;
    const double rtf_estimate = sim_advanced_s / std::max(1e-9, total_ms * 1e-3);

    if ((loop_count++ %
         std::max(1, static_cast<int>(cfg.control_rate_hz))) == 0) {
      RCLCPP_INFO(
          node->get_logger(),
          "[loop_timing] sim_t=%.3f mrt=%.2fms drake_substeps=%d "
          "step=%.2fms spin=%.2fms total=%.2fms budget=%.2fms "
          "overrun=%.2fms rtf_est=%.3f",
          adapter.getSimTime(),
          mrt_ms,
          physics_steps_per_control,
          step_ms,
          spin_ms,
          total_ms,
          budget_ms,
          overrun_ms,
          rtf_estimate);
    }

    if (overrun_ms > 0.0) {
      RCLCPP_WARN_THROTTLE(
          node->get_logger(),
          *node->get_clock(),
          1000,
          "[loop_overrun] total=%.2fms budget=%.2fms overrun=%.2fms "
          "mrt=%.2fms drake=%.2fms spin=%.2fms substeps=%d",
          total_ms,
          budget_ms,
          overrun_ms,
          mrt_ms,
          step_ms,
          spin_ms,
          physics_steps_per_control);
    }

    std::this_thread::sleep_until(next_tick);
  }

  rclcpp::shutdown();
  return 0;
}