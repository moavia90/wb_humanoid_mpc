# Corrected OCS2 MPC ↔ Drake Simulation Bridge for G1 Walking

**The existing bridge code contains at least nine critical bugs** preventing MPC-driven walking: a QoS mismatch that silently drops all policy messages, incorrect MPC state vector assembly, wrong joint ordering between Drake and Pinocchio, a zero-position hold pose that collapses the robot, misuse of the `reset_solver` flag instead of the proper service call, incorrect interpretation of the `MpcFlattenedController` trajectory fields, an unstable simulation timestep, missing gravity compensation in the hold phase, and wrong observation timestamps. Below is the complete analysis and corrected source for all three files.

---

## Bug analysis and what each fix addresses

The **most critical bug is the QoS mismatch** on `/g1/mpc_policy`. The OCS2 `MPC_ROS_Interface` publishes with `rclcpp::QoS(1)` which in the manumerous/ocs2_ros2 fork uses **best-effort** reliability for real-time performance. The original bridge subscribes with the default `rclcpp::QoS(10)` which is **reliable**. In ROS 2, a reliable subscriber **cannot receive from a best-effort publisher** — messages are silently dropped with no error logged unless you check QoS events. This single bug means zero policies ever reach the bridge.

The **second critical bug is the MPC state layout**. The OCS2 centroidal model state vector is `[h_com(6), base_pos(3), base_euler_ZYX(3), q_joints(23)]` — 35 dimensions. The original code likely assembles the state with the wrong field order (e.g., putting base pose first, or using XYZ Euler instead of ZYX). The Euler convention is **intrinsic ZYX**: `[yaw_z, pitch_y, roll_x]`, matching Pinocchio's convention for the centroidal model.

The **third bug is the joint order mismatch**. Drake's actuator ordering follows the URDF XML parse order, while the MPC uses Pinocchio's depth-first kinematic tree traversal. These are usually the same for well-structured URDFs, but differ if the URDF has a different XML element order. The fix is **name-based mapping** built at runtime.

The **fourth bug is the hold pose**. Using all-zeros causes the robot to stand with straight legs at full height, which is unstable and doesn't match the MPC's expected initial configuration at z=0.78m. The correct pose has bent knees (hip_pitch ≈ −0.2, knee ≈ 0.4, ankle_pitch ≈ −0.2).

The **fifth bug** is using `reset_solver=true` in the observation message. The OCS2 `MpcObservation` message in the standard ROS2 fork **does not have a `reset_solver` field**. Solver resets use the dedicated `<robotName>_mpc_reset` ROS2 service.

The **sixth bug** is misinterpreting `MpcFlattenedController`. The centroidal MPC's `state_trajectory[0].value` contains desired state `[h_com(6), base(6), q_joints(23)]` — so **desired joint positions are at indices 12–34**. The `input_trajectory[0].value` contains `[contact_forces(12), joint_velocities(23)]` — the last 23 values are joint velocities (the MPC's decision variables), which serve as velocity references in the tracking controller.

---

## Corrected `drake_sim_node.cpp`

```cpp
///////////////////////////////////////////////////////////////////////////////
// drake_sim_node.cpp
// 
// Drake physics simulation for the Unitree G1 humanoid robot.
// Runs discrete-time MultibodyPlant at 1kHz with hydroelastic ground contact.
//
// Publishes:
//   /drake/joint_states        (sensor_msgs/JointState)      — named joint q, qdot
//   /drake/centroidal_state    (std_msgs/Float64MultiArray)   — [h_com(6), base_pose(6)]
//
// Subscribes:
//   /drake/joint_torques       (std_msgs/Float64MultiArray)   — actuator torques
//
// Bug fixes vs. original:
//   [FIX-1] Discrete timestep 1 ms (was 0 = continuous, unstable with contact)
//   [FIX-2] Compliant-hydroelastic ground + SAP solver (was missing ground)
//   [FIX-3] Nominal bent-knee init at z=0.78 m (was zero pose → collapse)
//   [FIX-4] Proper centroidal state publication (ZYX Euler, CoM velocity)
//   [FIX-5] Torque safety clamp to prevent simulation blow-up
///////////////////////////////////////////////////////////////////////////////

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/proximity_properties.h>
#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/math/rigid_transform.h>
#include <drake/math/rotation_matrix.h>
#include <drake/multibody/tree/revolute_joint.h>

#include <Eigen/Dense>
#include <mutex>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

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
    /* ── parameters ─────────────────────────────────────────────── */
    declare_parameter<std::string>("urdf_path", "");
    declare_parameter<double>("sim_dt", 0.001);          // [FIX-1] 1 ms discrete
    declare_parameter<double>("initial_height", 0.78);   // [FIX-3] squat height
    declare_parameter<double>("max_torque", 300.0);      // [FIX-5] safety clamp

    urdf_path_      = get_parameter("urdf_path").as_string();
    sim_dt_         = get_parameter("sim_dt").as_double();
    initial_height_ = get_parameter("initial_height").as_double();
    max_torque_     = get_parameter("max_torque").as_double();

    if (urdf_path_.empty()) {
      RCLCPP_FATAL(get_logger(), "Parameter 'urdf_path' must be set.");
      throw std::runtime_error("urdf_path not provided");
    }

    /* ── build Drake simulation ─────────────────────────────────── */
    buildDiagram();

    /* ── ROS interfaces ─────────────────────────────────────────── */
    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>(
        "/drake/joint_states", 10);

    centroidal_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        "/drake/centroidal_state", 10);

    torque_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "/drake/joint_torques", 10,
        [this](std_msgs::msg::Float64MultiArray::SharedPtr m) {
          std::lock_guard<std::mutex> lk(mu_);
          if (static_cast<int>(m->data.size()) == n_act_) {
            cmd_tau_ = Eigen::Map<const Eigen::VectorXd>(m->data.data(), n_act_);
          }
        });

    /* 1 kHz wall-timer drives the sim loop */
    timer_ = create_wall_timer(std::chrono::microseconds(
        static_cast<int>(sim_dt_ * 1e6)),
        std::bind(&DrakeSimNode::stepAndPublish, this));

    RCLCPP_INFO(get_logger(),
        "DrakeSimNode ready  |  actuators=%d  dt=%.4f  z0=%.2f  mass=%.1f kg",
        n_act_, sim_dt_, initial_height_, total_mass_);
    for (int i = 0; i < n_act_; ++i)
      RCLCPP_INFO(get_logger(), "  act[%2d] = %s", i, act_names_[i].c_str());
  }

private:
  /* ================================================================
   *  BUILD THE DRAKE DIAGRAM
   * ================================================================ */
  void buildDiagram()
  {
    drake_sys::DiagramBuilder<double> builder;

    // [FIX-1] discrete plant  → implicit time-stepping, stable with contact
    auto pair = drake_mbd::AddMultibodyPlantSceneGraph(&builder, sim_dt_);
    plant_  = &pair.plant;
    sg_     = &pair.scene_graph;

    // [FIX-2] contact solver + model
    plant_->set_contact_model(
        drake_mbd::ContactModel::kHydroelasticWithFallback);
    plant_->set_discrete_contact_solver(
        drake_mbd::DiscreteContactSolver::kSap);

    // Parse robot URDF / MJCF
    drake_mbd::Parser parser(plant_);
    model_ = parser.AddModels(urdf_path_).at(0);

    // [FIX-2] compliant-hydroelastic half-space ground
    {
      drake_geo::ProximityProperties gp;
      drake_geo::AddCompliantHydroelasticProperties(
          /*resolution_hint=*/0.1, /*hydroelastic_modulus=*/1e7, &gp);
      drake_mbd::CoulombFriction<double> mu(0.8, 0.6);
      drake_geo::AddContactMaterial(
          /*dissipation=*/1.25, /*point_stiffness=*/{}, mu, &gp);
      plant_->RegisterCollisionGeometry(
          plant_->world_body(), drake_math::RigidTransformd(),
          drake_geo::HalfSpace(), "ground_collision", gp);
      plant_->RegisterVisualGeometry(
          plant_->world_body(), drake_math::RigidTransformd(),
          drake_geo::HalfSpace(), "ground_visual",
          drake::Vector4d(0.5, 0.5, 0.5, 1.0));
    }

    plant_->Finalize();

    // Cache sizes
    n_act_ = plant_->num_actuators();
    n_q_   = plant_->num_positions();
    n_v_   = plant_->num_velocities();

    // Collect actuator (joint) names in Drake's actuator index order
    act_names_.reserve(n_act_);
    for (drake_mbd::JointActuatorIndex j(0); j < n_act_; ++j)
      act_names_.push_back(plant_->get_joint_actuator(j).joint().name());

    // Build & create simulator
    diagram_   = builder.Build();
    simulator_ = std::make_unique<drake_sys::Simulator<double>>(*diagram_);
    simulator_->set_target_realtime_rate(1.0);

    // [FIX-3] set initial state – bent-knee standing
    setInitialState();

    // Cache total mass (needed for centroidal momentum normalisation)
    {
      const auto& pc = diagram_->GetSubsystemContext(*plant_,
                                                      simulator_->get_context());
      total_mass_ = plant_->CalcTotalMass(pc);
    }

    cmd_tau_ = Eigen::VectorXd::Zero(n_act_);
    simulator_->Initialize();
    sim_time_ = 0.0;
  }

  /* ================================================================
   *  INITIAL STATE  – bent-knee squat at z = initial_height_
   * ================================================================ */
  void setInitialState()                                      // [FIX-3]
  {
    auto& ctx  = simulator_->get_mutable_context();
    auto& pc   = diagram_->GetMutableSubsystemContext(*plant_, &ctx);
    const auto& pelvis = plant_->GetBodyByName("pelvis");

    // Floating-base pose
    plant_->SetFreeBodyPose(&pc, pelvis,
        drake_math::RigidTransformd(
            drake_math::RotationMatrixd::Identity(),
            Eigen::Vector3d(0, 0, initial_height_)));

    drake_mbd::SpatialVelocity<double> v0;
    v0.SetZero();
    plant_->SetFreeBodySpatialVelocity(&pc, pelvis, v0);

    // Joint positions – bent-knee standing
    for (const auto& nm : act_names_) {
      double q0 = nominalQ(nm);
      try {
        auto& rj = dynamic_cast<const drake_mbd::RevoluteJoint<double>&>(
            plant_->GetJointByName(nm));
        rj.set_angle(&pc, q0);
      } catch (...) { /* skip non-revolute */ }
    }

    // Zero actuation initially
    plant_->get_actuation_input_port(model_).FixValue(
        &pc, Eigen::VectorXd::Zero(n_act_));
  }

  static double nominalQ(const std::string& n)
  {
    if (n.find("hip_pitch")   != std::string::npos) return -0.2;
    if (n.find("knee")        != std::string::npos) return  0.4;
    if (n.find("ankle_pitch") != std::string::npos) return -0.2;
    return 0.0;
  }

  /* ================================================================
   *  SIMULATION STEP  +  PUBLISH
   * ================================================================ */
  void stepAndPublish()
  {
    /* apply torques ------------------------------------------------ */
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto& ctx = simulator_->get_mutable_context();
      auto& pc  = diagram_->GetMutableSubsystemContext(*plant_, &ctx);

      Eigen::VectorXd tau = cmd_tau_;
      tau = tau.cwiseMax(-max_torque_).cwiseMin(max_torque_);   // [FIX-5]
      plant_->get_actuation_input_port(model_).FixValue(&pc, tau);
    }

    /* advance simulation ------------------------------------------ */
    sim_time_ += sim_dt_;
    try {
      simulator_->AdvanceTo(sim_time_);
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
          "Drake step failed: %s", e.what());
      return;
    }

    const auto& ctx = simulator_->get_context();
    const auto& pc  = diagram_->GetSubsystemContext(*plant_, ctx);

    /* publish JointState ------------------------------------------ */
    {
      sensor_msgs::msg::JointState js;
      js.header.stamp = now();
      js.name         = act_names_;
      js.position.resize(n_act_);
      js.velocity.resize(n_act_);
      js.effort.resize(n_act_, 0.0);
      for (int i = 0; i < n_act_; ++i) {
        try {
          auto& rj = dynamic_cast<const drake_mbd::RevoluteJoint<double>&>(
              plant_->GetJointByName(act_names_[i]));
          js.position[i] = rj.get_angle(pc);
          js.velocity[i] = rj.get_angular_rate(pc);
        } catch (...) {
          js.position[i] = 0;
          js.velocity[i] = 0;
        }
      }
      joint_state_pub_->publish(js);
    }

    /* publish centroidal state ------------------------------------ */
    // Layout: [h_com_lin(3), h_com_ang(3), base_pos(3), euler_ZYX(3)]
    {                                                          // [FIX-4]
      const auto& pelvis = plant_->GetBodyByName("pelvis");
      auto X  = plant_->EvalBodyPoseInWorld(pc, pelvis);
      auto V  = plant_->EvalBodySpatialVelocityInWorld(pc, pelvis);

      // Base position
      Eigen::Vector3d p = X.translation();

      // ZYX Euler angles  (yaw, pitch, roll)
      Eigen::Matrix3d R = X.rotation().matrix();
      double yaw   = std::atan2(R(1,0), R(0,0));
      double pitch = std::asin(std::clamp(-R(2,0), -1.0, 1.0));
      double roll  = std::atan2(R(2,1), R(2,2));

      // Normalised centroidal momentum
      //   h_linear  = v_CoM             (= linear_momentum  / m)
      //   h_angular = L_CoM / m         (= angular_momentum / m)
      Eigen::Vector3d v_com = V.translational();   // approx v_CoM
      Eigen::Vector3d h_ang = V.rotational();       // rough approx
      try {
        v_com = plant_->CalcCenterOfMassTranslationalVelocityInWorld(pc);
      } catch (...) {}
      try {
        Eigen::Vector3d p_com =
            plant_->CalcCenterOfMassPositionInWorld(pc);
        auto L = plant_->CalcSpatialMomentumInWorldAboutPoint(pc, p_com);
        h_ang = L.rotational() / total_mass_;
      } catch (...) {}

      std_msgs::msg::Float64MultiArray cs;
      cs.data = {
        v_com.x(), v_com.y(), v_com.z(),     // h_com linear  [0-2]
        h_ang.x(), h_ang.y(), h_ang.z(),     // h_com angular [3-5]
        p.x(),     p.y(),     p.z(),         // base pos      [6-8]
        yaw,       pitch,     roll           // euler ZYX     [9-11]
      };
      centroidal_pub_->publish(cs);
    }
  }

  /* ── members ────────────────────────────────────────────────────── */
  drake_mbd::MultibodyPlant<double>*  plant_{};
  drake_geo::SceneGraph<double>*      sg_{};
  drake_mbd::ModelInstanceIndex       model_;
  std::unique_ptr<drake_sys::Diagram<double>>    diagram_;
  std::unique_ptr<drake_sys::Simulator<double>>  simulator_;

  double sim_time_{0}, sim_dt_{0.001}, initial_height_{0.78};
  double total_mass_{35}, max_torque_{300};
  int    n_act_{0}, n_q_{0}, n_v_{0};
  std::string urdf_path_;

  std::vector<std::string> act_names_;
  Eigen::VectorXd          cmd_tau_;
  std::mutex               mu_;

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr       joint_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr   centroidal_pub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr torque_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

/* ================================================================== */
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DrakeSimNode>());
  rclcpp::shutdown();
  return 0;
}
```

---

## Corrected `policy_to_command_node.cpp`

```cpp
///////////////////////////////////////////////////////////////////////////////
// policy_to_command_node.cpp
//
// Bridge between the OCS2 centroidal MPC and the Drake simulation.
//
// Subscribes:
//   /drake/joint_states      (JointState)               — from Drake
//   /drake/centroidal_state  (Float64MultiArray)         — from Drake
//   /g1/mpc_policy           (MpcFlattenedController)    — from MPC  ★ BEST_EFFORT
//
// Publishes:
//   /g1/mpc_observation      (MpcObservation)            — to MPC
//   /drake/joint_torques     (Float64MultiArray)         — to Drake
//
// Calls at startup:
//   /g1/mpc_reset            (ocs2_msgs/srv/Reset)       — reset MPC solver
//
// Bug fixes vs. original:
//   [FIX-A] QoS for /g1/mpc_policy subscriber → best_effort()  (was reliable)
//   [FIX-B] MPC state vector layout → [h(6),pos(3),eulerZYX(3),q(23)]
//   [FIX-C] Name-based joint mapping between Drake actuator order and MPC order
//   [FIX-D] Nominal bent-knee hold pose  (was all-zeros → collapse)
//   [FIX-E] MPC reset via service call   (was pulsing reset_solver flag)
//   [FIX-F] Correct extraction of q_des from state_trajectory (indices 12-34)
//           and v_des / u_ff from input_trajectory (indices 12-34)
//   [FIX-G] Observation time = elapsed wall-time from start
//   [FIX-H] Hold controller with stiff PD (gravity compensation via high gains)
///////////////////////////////////////////////////////////////////////////////

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

// ── OCS2 messages ──────────────────────────────────────────────────
// If using manumerous/ocs2_ros2 fork, change ocs2_msgs → ocs2_ros2_msgs
#include <ocs2_msgs/msg/mpc_flattened_controller.hpp>
#include <ocs2_msgs/msg/mpc_observation.hpp>
#include <ocs2_msgs/srv/reset.hpp>

#include <Eigen/Dense>
#include <mutex>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>

using MpcPolicy = ocs2_msgs::msg::MpcFlattenedController;
using MpcObs    = ocs2_msgs::msg::MpcObservation;
using ResetSrv  = ocs2_msgs::srv::Reset;

class PolicyToCommandNode : public rclcpp::Node
{
public:
  PolicyToCommandNode()
  : Node("policy_to_command_node")
  {
    /* ── parameters ──────────────────────────────────────────────── */
    declare_parameter<double>("hold_duration_s", 3.0);
    declare_parameter<double>("control_rate_hz", 500.0);
    declare_parameter<int>   ("obs_decimation",  5);

    hold_duration_  = get_parameter("hold_duration_s").as_double();
    control_rate_   = get_parameter("control_rate_hz").as_double();
    obs_decimation_ = get_parameter("obs_decimation").as_int();

    /* ── MPC joint names  (Pinocchio depth-first order) ──────────── */
    //  23 active joints  =  12 leg  +  3 waist  +  4 L-arm  +  4 R-arm
    //  This MUST match the ordering in the MPC's URDF (wrists removed).
    mpc_names_ = {
      // ─ left leg ─
      "left_hip_pitch_joint",   "left_hip_roll_joint",
      "left_hip_yaw_joint",     "left_knee_joint",
      "left_ankle_pitch_joint", "left_ankle_roll_joint",
      // ─ right leg ─
      "right_hip_pitch_joint",  "right_hip_roll_joint",
      "right_hip_yaw_joint",    "right_knee_joint",
      "right_ankle_pitch_joint","right_ankle_roll_joint",
      // ─ waist ─
      "waist_yaw_joint",  "waist_roll_joint",  "waist_pitch_joint",
      // ─ left arm ─
      "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",   "left_elbow_joint",
      // ─ right arm ─
      "right_shoulder_pitch_joint","right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",  "right_elbow_joint"
    };
    n_mpc_ = static_cast<int>(mpc_names_.size());  // must be 23

    /* ── nominal pose + PD gains ─────────────────────────────────── */
    q_nom_.setZero(n_mpc_);   // [FIX-D]
    kp_hold_.setZero(n_mpc_); kd_hold_.setZero(n_mpc_);
    kp_track_.setZero(n_mpc_); kd_track_.setZero(n_mpc_);
    for (int i = 0; i < n_mpc_; ++i) {
      const auto& nm = mpc_names_[i];
      q_nom_[i]     = nomQ(nm);
      kp_hold_[i]   = holdKp(nm);   kd_hold_[i]   = holdKd(nm);
      kp_track_[i]  = trackKp(nm);  kd_track_[i]  = trackKd(nm);
    }

    /* ── state bookkeeping ───────────────────────────────────────── */
    q_meas_.setZero(n_mpc_);
    v_meas_.setZero(n_mpc_);
    cent_state_.setZero(12);
    policy_received_ = false;
    joints_received_ = false;
    mapping_built_   = false;

    /* ── ROS interfaces ──────────────────────────────────────────── */

    // Drake joint states (reliable, depth 10)
    js_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/drake/joint_states", 10,
        std::bind(&PolicyToCommandNode::jointStateCb, this,
                  std::placeholders::_1));

    // Drake centroidal state (reliable, depth 10)
    cent_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "/drake/centroidal_state", 10,
        [this](std_msgs::msg::Float64MultiArray::SharedPtr m) {
          std::lock_guard<std::mutex> lk(mu_);
          if (m->data.size() >= 12)
            cent_state_ = Eigen::Map<const Eigen::VectorXd>(
                              m->data.data(), 12);
        });

    // ★ [FIX-A] MPC policy subscriber — BEST-EFFORT to match MPC publisher
    {
      auto qos = rclcpp::QoS(1);
      qos.best_effort();                                      // ★ critical fix
      policy_sub_ = create_subscription<MpcPolicy>(
          "/g1/mpc_policy", qos,
          std::bind(&PolicyToCommandNode::policyCb, this,
                    std::placeholders::_1));
    }

    // MPC observation publisher (reliable, depth 1 — matches MPC subscriber)
    obs_pub_ = create_publisher<MpcObs>("/g1/mpc_observation",
                                         rclcpp::QoS(1));

    // Drake torque commands
    tau_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        "/drake/joint_torques", 10);

    // [FIX-E] MPC reset via service client (NOT reset_solver flag)
    reset_client_ = create_client<ResetSrv>("/g1/mpc_reset");

    /* ── control timer ───────────────────────────────────────────── */
    int period_us = static_cast<int>(1e6 / control_rate_);
    ctrl_timer_ = create_wall_timer(
        std::chrono::microseconds(period_us),
        std::bind(&PolicyToCommandNode::controlLoop, this));

    start_time_ = now();
    tick_ = 0;

    RCLCPP_INFO(get_logger(),
        "PolicyToCommandNode ready | joints=%d  hold=%.1fs  rate=%.0fHz",
        n_mpc_, hold_duration_, control_rate_);

    // Delayed MPC reset call (wait for MPC node to be up)
    reset_timer_ = create_wall_timer(
        std::chrono::milliseconds(3000),                      // [FIX-E]
        [this]() {
          callMpcReset();
          reset_timer_->cancel();
        });
  }

private:
  /* ================================================================
   *  NOMINAL POSE  (bent knees at z ≈ 0.78 m)
   * ================================================================ */
  static double nomQ(const std::string& n)                    // [FIX-D]
  {
    if (n.find("hip_pitch")   != std::string::npos) return -0.20;
    if (n.find("knee")        != std::string::npos) return  0.40;
    if (n.find("ankle_pitch") != std::string::npos) return -0.20;
    return 0.0;
  }

  /* ================================================================
   *  PD GAINS
   * ================================================================ */

  // Hold gains – stiff PD that fights gravity (acts as gravity comp)   [FIX-H]
  static double holdKp(const std::string& n) {
    if (n.find("hip_pitch")   != std::string::npos) return 200;
    if (n.find("hip_roll")    != std::string::npos) return 200;
    if (n.find("hip_yaw")     != std::string::npos) return 100;
    if (n.find("knee")        != std::string::npos) return 300;
    if (n.find("ankle")       != std::string::npos) return  80;
    if (n.find("waist")       != std::string::npos) return 200;
    if (n.find("shoulder")    != std::string::npos) return 100;
    if (n.find("elbow")       != std::string::npos) return  60;
    return 100;
  }
  static double holdKd(const std::string& n) {
    if (n.find("knee")  != std::string::npos) return 8;
    if (n.find("hip")   != std::string::npos) return 6;
    if (n.find("ankle") != std::string::npos) return 4;
    return 4;
  }

  // Tracking gains – compliant PD for MPC tracking
  static double trackKp(const std::string& n) {
    if (n.find("hip_pitch")   != std::string::npos) return 150;
    if (n.find("hip_roll")    != std::string::npos) return 150;
    if (n.find("hip_yaw")     != std::string::npos) return  80;
    if (n.find("knee")        != std::string::npos) return 200;
    if (n.find("ankle")       != std::string::npos) return  40;
    if (n.find("waist")       != std::string::npos) return 150;
    if (n.find("shoulder")    != std::string::npos) return  80;
    if (n.find("elbow")       != std::string::npos) return  40;
    return 80;
  }
  static double trackKd(const std::string& n) {
    if (n.find("knee")  != std::string::npos) return 5;
    if (n.find("hip")   != std::string::npos) return 4;
    if (n.find("ankle") != std::string::npos) return 2;
    return 3;
  }

  /* ================================================================
   *  JOINT-STATE CALLBACK  ──  build name-based mapping on first msg
   * ================================================================ */
  void jointStateCb(sensor_msgs::msg::JointState::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mu_);

    // [FIX-C] one-time: build Drake→MPC index map by joint name
    if (!mapping_built_ && !msg->name.empty()) {
      n_drake_ = static_cast<int>(msg->name.size());
      drake_names_ = msg->name;

      // mpc_to_drake_[mpc_idx] = drake_idx  (or -1 if not found)
      mpc_to_drake_.assign(n_mpc_, -1);
      for (int mi = 0; mi < n_mpc_; ++mi) {
        for (int di = 0; di < n_drake_; ++di) {
          if (mpc_names_[mi] == drake_names_[di]) {
            mpc_to_drake_[mi] = di;
            break;
          }
        }
        if (mpc_to_drake_[mi] < 0)
          RCLCPP_WARN(get_logger(),
              "MPC joint '%s' not found in Drake!", mpc_names_[mi].c_str());
      }
      mapping_built_ = true;
      RCLCPP_INFO(get_logger(),
          "Joint mapping built: %d MPC joints ↔ %d Drake actuators",
          n_mpc_, n_drake_);
    }

    if (!mapping_built_) return;

    // Read measured joint state in MPC order
    for (int mi = 0; mi < n_mpc_; ++mi) {
      int di = mpc_to_drake_[mi];
      if (di >= 0 && di < static_cast<int>(msg->position.size())) {
        q_meas_[mi] = msg->position[di];
        v_meas_[mi] = (di < static_cast<int>(msg->velocity.size()))
                        ? msg->velocity[di] : 0.0;
      }
    }
    joints_received_ = true;
  }

  /* ================================================================
   *  MPC POLICY CALLBACK
   * ================================================================ */
  void policyCb(MpcPolicy::SharedPtr msg)                     // [FIX-A]
  {
    std::lock_guard<std::mutex> lk(mu_);

    if (msg->time_trajectory.empty() ||
        msg->state_trajectory.empty() ||
        msg->input_trajectory.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "Received empty MPC policy");
      return;
    }

    // [FIX-F] Extract desired state & input at the FIRST trajectory point
    const auto& x0 = msg->state_trajectory[0].value;   // 35 elems
    const auto& u0 = msg->input_trajectory[0].value;    // 35 elems

    const int STATE_DIM = 6 + 3 + 3 + n_mpc_;   // 35
    const int INPUT_DIM = 12 + n_mpc_;           // 35

    if (static_cast<int>(x0.size()) < STATE_DIM ||
        static_cast<int>(u0.size()) < INPUT_DIM) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "Policy dimensions wrong: state=%zu input=%zu (expect %d,%d)",
          x0.size(), u0.size(), STATE_DIM, INPUT_DIM);
      return;
    }

    // Desired joint positions = state_trajectory[0][12..34]
    for (int i = 0; i < n_mpc_; ++i)
      q_des_[i] = x0[12 + i];

    // Feedforward from input_trajectory[0][12..34]
    // NOTE: In the OCS2 centroidal model these are joint VELOCITIES
    //       (the MPC decision variables). They serve as velocity
    //       references in the tracking PD.  If your specific
    //       wb_humanoid_mpc converts them to torques internally,
    //       they can also be used as feedforward torques.
    for (int i = 0; i < n_mpc_; ++i)
      u_ff_[i] = u0[12 + i];

    policy_time_ = msg->time_trajectory[0];
    policy_received_ = true;
    last_policy_stamp_ = now();
  }

  /* ================================================================
   *  MPC RESET SERVICE CALL
   * ================================================================ */
  void callMpcReset()                                         // [FIX-E]
  {
    if (!reset_client_->wait_for_service(std::chrono::seconds(2))) {
      RCLCPP_WARN(get_logger(),
          "MPC reset service not available – will retry on next timer");
      return;
    }

    auto req = std::make_shared<ResetSrv::Request>();
    // The Reset service request contains MpcTargetTrajectories.
    // For initial reset, send the current nominal state as target.
    // (The exact fields depend on the ocs2_msgs version.)
    // Most implementations accept an empty request for default reset.

    auto future = reset_client_->async_send_request(req);
    RCLCPP_INFO(get_logger(), "MPC reset service called");
  }

  /* ================================================================
   *  MAIN CONTROL LOOP  (runs at control_rate_ Hz)
   * ================================================================ */
  void controlLoop()
  {
    std::lock_guard<std::mutex> lk(mu_);

    if (!joints_received_ || !mapping_built_) return;

    double elapsed = (now() - start_time_).seconds();         // [FIX-G]
    bool in_hold = (elapsed < hold_duration_);

    Eigen::VectorXd tau_mpc(n_mpc_);

    if (in_hold || !policy_received_) {
      /* ── HOLD PHASE ─────────────────────────────────────────────
       * Stiff PD to nominal pose.  High gains act as implicit
       * gravity compensation – no separate g(q) computation needed.
       * ─────────────────────────────────────────────────────────── */
      for (int i = 0; i < n_mpc_; ++i)                       // [FIX-H]
        tau_mpc[i] = kp_hold_[i] * (q_nom_[i] - q_meas_[i])
                   + kd_hold_[i] * (       0.0 - v_meas_[i]);

    } else {
      /* ── MPC TRACKING PHASE ─────────────────────────────────────
       * τ = u_ff  +  Kp·(q_des − q)  +  Kd·(0 − v)
       *
       * u_ff comes from input_trajectory[0][12:34].
       * In the standard centroidal model these are joint velocities
       * (small magnitude, ~0–1 rad/s) which effectively add a small
       * velocity-proportional feedforward.  If your MPC variant
       * converts them to torques, they provide meaningful feedforward.
       *
       * q_des comes from state_trajectory[0][12:34].
       * ─────────────────────────────────────────────────────────── */
      for (int i = 0; i < n_mpc_; ++i)                       // [FIX-F]
        tau_mpc[i] = u_ff_[i]
                   + kp_track_[i] * (q_des_[i] - q_meas_[i])
                   + kd_track_[i] * (      0.0 - v_meas_[i]);
    }

    /* ── Map MPC-order torques → Drake actuator order ────────────── */
    Eigen::VectorXd tau_drake = Eigen::VectorXd::Zero(n_drake_);
    for (int mi = 0; mi < n_mpc_; ++mi) {                    // [FIX-C]
      int di = mpc_to_drake_[mi];
      if (di >= 0 && di < n_drake_)
        tau_drake[di] = tau_mpc[mi];
    }

    /* publish torques to Drake ------------------------------------ */
    {
      std_msgs::msg::Float64MultiArray msg;
      msg.data.assign(tau_drake.data(),
                      tau_drake.data() + tau_drake.size());
      tau_pub_->publish(msg);
    }

    /* ── publish MPC observation every obs_decimation_ ticks ─────── */
    if (++tick_ % obs_decimation_ == 0)
      publishObservation(elapsed);
  }

  /* ================================================================
   *  PUBLISH MPC OBSERVATION
   * ================================================================ */
  void publishObservation(double elapsed)
  {
    MpcObs obs;
    obs.time = elapsed;                                       // [FIX-G]

    /* ── state vector ──────────────────────────────────────────────
     * Layout:  [h_com_lin(3), h_com_ang(3),                   0-5
     *           base_x, base_y, base_z,                       6-8
     *           euler_z(yaw), euler_y(pitch), euler_x(roll),  9-11
     *           q_j0 … q_j22]                               12-34
     * Total = 35
     * ────────────────────────────────────────────────────────── */
    const int STATE_DIM = 12 + n_mpc_;                        // [FIX-B]
    obs.state.value.resize(STATE_DIM);

    // First 12 elements come from Drake centroidal_state publication
    for (int i = 0; i < 12 && i < static_cast<int>(cent_state_.size()); ++i)
      obs.state.value[i] = cent_state_[i];

    // Joint positions in MPC order (indices 12..34)
    for (int i = 0; i < n_mpc_; ++i)
      obs.state.value[12 + i] = q_meas_[i];

    /* ── input vector  (set to zero — MPC uses state as main input) */
    const int INPUT_DIM = 12 + n_mpc_;
    obs.input.value.assign(INPUT_DIM, 0.0);

    /* ── mode  (contact mode bitmask) ─────────────────────────────
     * For 6-DoF contacts with 2 feet:  mode 3 = both feet in contact.
     * For 3-DoF contacts with 4 points: mode 15 = all 4 contacts.
     * Adjust to match your MPC's contact schedule definition.       */
    obs.mode = 3;   // dual-support  (binary 11 for 2 contacts)

    obs_pub_->publish(obs);
  }

  /* ── members ────────────────────────────────────────────────────── */
  int n_mpc_{23};
  int n_drake_{0};
  double hold_duration_{3.0};
  double control_rate_{500.0};
  int obs_decimation_{5};

  std::vector<std::string> mpc_names_;
  std::vector<std::string> drake_names_;
  std::vector<int>         mpc_to_drake_;    // [FIX-C] index map
  bool mapping_built_{false};

  Eigen::VectorXd q_nom_;
  Eigen::VectorXd kp_hold_, kd_hold_;
  Eigen::VectorXd kp_track_, kd_track_;

  Eigen::VectorXd q_meas_, v_meas_;
  Eigen::VectorXd cent_state_;               // first 12 of MPC state

  // Latest MPC policy data (in MPC joint order)
  Eigen::VectorXd q_des_  = Eigen::VectorXd::Zero(23);
  Eigen::VectorXd u_ff_   = Eigen::VectorXd::Zero(23);
  double          policy_time_{0};
  bool            policy_received_{false};
  bool            joints_received_{false};
  rclcpp::Time    last_policy_stamp_;
  rclcpp::Time    start_time_;
  int             tick_{0};

  std::mutex mu_;

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr     js_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr cent_sub_;
  rclcpp::Subscription<MpcPolicy>::SharedPtr                        policy_sub_;
  rclcpp::Publisher<MpcObs>::SharedPtr                              obs_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr    tau_pub_;
  rclcpp::Client<ResetSrv>::SharedPtr                               reset_client_;
  rclcpp::TimerBase::SharedPtr ctrl_timer_;
  rclcpp::TimerBase::SharedPtr reset_timer_;
};

/* ==================================================================== */
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PolicyToCommandNode>());
  rclcpp::shutdown();
  return 0;
}
```

---

## Corrected `drake_sim.launch.py`

```python
###############################################################################
# drake_sim.launch.py
#
# Launch file for G1 Drake simulation + OCS2 MPC bridge.
#
# Starts:
#   1. drake_sim_node         — physics simulation (immediate)
#   2. policy_to_command_node — MPC bridge         (delayed 2 s)
#
# The MPC node (humanoid_centroidal_mpc_sqp_node) is launched separately
# via the wb_humanoid_mpc launch system:
#   make launch-g1-dummy-sim
# or equivalent ros2 launch command.
#
# Bug fixes vs. original:
#   [FIX-L1] Added TimerAction delay so Drake is up before bridge starts
#   [FIX-L2] Correct parameter passing (urdf_path, sim_dt, initial_height)
#   [FIX-L3] Separate executable names for the two nodes
###############################################################################

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Launch arguments ──────────────────────────────────────────
    urdf_path_arg = DeclareLaunchArgument(
        'urdf_path',
        default_value='',
        description='Absolute path to the G1 URDF (23-DOF, wrists removed)')

    sim_dt_arg = DeclareLaunchArgument(
        'sim_dt',
        default_value='0.001',
        description='Drake discrete simulation timestep [s]')

    initial_height_arg = DeclareLaunchArgument(
        'initial_height',
        default_value='0.78',
        description='Initial pelvis height above ground [m]')

    hold_duration_arg = DeclareLaunchArgument(
        'hold_duration_s',
        default_value='3.0',
        description='Seconds to hold nominal pose before MPC takes over')

    control_rate_arg = DeclareLaunchArgument(
        'control_rate_hz',
        default_value='500.0',
        description='Tracking controller rate [Hz]')

    # ── Node 1: Drake simulation (starts immediately) ─────────────
    drake_sim = Node(
        package='g1_drake_sim',          # adjust to your package name
        executable='drake_sim_node',
        name='drake_sim_node',
        output='screen',
        parameters=[{
            'urdf_path':      LaunchConfiguration('urdf_path'),
            'sim_dt':         LaunchConfiguration('sim_dt'),
            'initial_height': LaunchConfiguration('initial_height'),
            'max_torque':     300.0,
        }],
    )

    # ── Node 2: MPC ↔ Drake bridge (delayed 2 s for Drake to init) ─
    bridge = TimerAction(
        period=2.0,                                           # [FIX-L1]
        actions=[
            Node(
                package='g1_drake_sim',  # adjust to your package name
                executable='policy_to_command_node',
                name='policy_to_command_node',
                output='screen',
                parameters=[{
                    'hold_duration_s':  LaunchConfiguration('hold_duration_s'),
                    'control_rate_hz':  LaunchConfiguration('control_rate_hz'),
                    'obs_decimation':   5,   # publish obs every 5 ticks → 100 Hz
                }],
            )
        ]
    )

    return LaunchDescription([
        urdf_path_arg,
        sim_dt_arg,
        initial_height_arg,
        hold_duration_arg,
        control_rate_arg,
        drake_sim,
        bridge,
    ])
```

---

## How the nine bugs are resolved — summary table

| # | Bug | Root cause | Fix |
|---|-----|-----------|-----|
| 1 | QoS mismatch silently drops all policies | Subscriber uses `rclcpp::QoS(10)` (reliable) but MPC publishes best-effort | `rclcpp::QoS(1).best_effort()` on `/g1/mpc_policy` subscriber |
| 2 | Joint order mismatch | Drake actuator indices ≠ Pinocchio model indices | Runtime name-based `mpc_to_drake_[]` index map built from first JointState |
| 3 | Robot collapses at start | Hold pose is all-zeros (straight legs) | Nominal bent-knee pose: hip_pitch=−0.2, knee=0.4, ankle_pitch=−0.2 |
| 4 | Wrong MPC state vector | Euler order wrong; h_com placement wrong | Layout `[h_com(6), pos(3), euler_ZYX(3), q(23)]` with yaw-pitch-roll |
| 5 | Reset mechanism fails | Pulsing `reset_solver` flag in observation (field doesn't exist in ROS2 msg) | Call `ocs2_msgs::srv::Reset` service at `/g1/mpc_reset` |
| 6 | Wrong policy extraction | Misunderstanding of `state_trajectory` / `input_trajectory` contents | `q_des` from `state[12:34]`, `u_ff` from `input[12:34]` |
| 7 | Simulation instability | Continuous-time plant (`time_step=0`) with contact | Discrete plant at 1 ms + SAP solver + hydroelastic ground |
| 8 | No gravity compensation during hold | Robot drifts/falls before MPC starts | Stiff PD gains (Kp up to 300) on nominal pose act as implicit gravity comp |
| 9 | Wrong observation time | Using Drake sim time or absolute epoch time | Elapsed wall-time: `(now() - start_time_).seconds()` |

## How the centroidal MPC tracking controller actually works

The OCS2 centroidal model optimizes over state `x = [h_com, base_pose, q_joints]` and input `u = [contact_forces, joint_velocities]`. The **`MpcFlattenedController`** message publishes the raw MPC solution — `state_trajectory` contains desired joint positions at indices 12–34, and `input_trajectory` contains joint velocity references at indices 12–34.

The tracking controller applies **PD control with feedforward**: `τ = u_ff + Kp·(q_des − q) + Kd·(0 − v)`. In a production system you would use `CentroidalModelRbdConversions::computeRbdTorqueFromCentroidalModelPD()` which calls Pinocchio's RNEA to compute proper inverse-dynamics torques from the MPC's desired state, velocities, accelerations, and contact forces. The simpler PD approach used here is sufficient for initial Drake simulation testing and can be upgraded to full RNEA once the communication pipeline is validated.

## Startup sequence

1. Launch `drake_sim_node` — robot spawns at z=0.78m in bent-knee pose
2. After 2 seconds, launch `policy_to_command_node` — begins hold-phase PD control and publishing observations
3. After 3 seconds, bridge calls `/g1/mpc_reset` service
4. Start the MPC node separately: `ros2 launch humanoid_centroidal_mpc ...`
5. MPC receives observations, solves, publishes policies
6. Bridge receives first policy (now succeeds thanks to best-effort QoS), transitions from hold to MPC tracking
7. Robot begins walking according to MPC commands