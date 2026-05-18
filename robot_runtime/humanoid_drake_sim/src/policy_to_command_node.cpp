///////////////////////////////////////////////////////////////////////////////
// policy_to_command_node.cpp  —  Drake centroidal MPC bridge
//
// Uses the actual OCS2 workspace helpers:
//   - computeJointTorques<double>(...) from humanoid_common_mpc
//   - stanceLeg2ModeNumber(...) from humanoid_common_mpc
//   - createCustomPinocchioInterface(...) from humanoid_common_mpc
//
// weight‑compensating wrenches are computed manually (no extra header).
///////////////////////////////////////////////////////////////////////////////

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_ros2_msgs/msg/mpc_flattened_controller.hpp>
#include <ocs2_ros2_msgs/msg/mpc_observation.hpp>
#include <ocs2_ros2_msgs/msg/mpc_target_trajectories.hpp>
#include <ocs2_ros2_msgs/srv/reset.hpp>

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>

// Safe headers: do NOT include MpcRobotModelBase.h
// (which pulls in boost/property_tree → boost/multi_index, broken with BOOST_MPL_LIMIT=30)
#include <humanoid_common_mpc/common/ModelSettings.h>
#include <humanoid_common_mpc/common/Types.h>
#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>
#include <humanoid_common_mpc/pinocchio_model/createPinocchioModel.h>
// Raw Pinocchio algorithms — no MpcRobotModelBase dependency
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
// pinocchio::nonLinearEffects() is in compute-all-terms.hpp in this Pinocchio version.
// (nonLinearEffects.hpp was renamed/merged; compute-all-terms.hpp populates data.nle)
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/fwd.hpp>


#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using MpcPolicy   = ocs2_ros2_msgs::msg::MpcFlattenedController;
using MpcObsMsg   = ocs2_ros2_msgs::msg::MpcObservation;
using ResetSrv    = ocs2_ros2_msgs::srv::Reset;

namespace ocs2::humanoid {

namespace {
constexpr double kGlobalTauLimit = 180.0;
constexpr double kDefaultMpcFeedforwardScale = 0.0;
// Safe default: validate standing with MPC feedforward disabled first.
// Re-enable gradually (0.05 -> 0.10 -> 0.15) after the robot can stand
// for 60s after startup-pin release.
constexpr double kHoldFeedforwardScale = 1.0;
constexpr double kPolicyLookahead = 0.005;
constexpr double kContactForceThreshold = 10.0;
constexpr double kGravity = 9.81;
constexpr double kMinMpcPelvisZ = 0.55;
constexpr double kMaxMpcPelvisZ = 1.30;
constexpr double kMinResetPelvisZ = 0.70;
constexpr double kMaxResetPelvisZ = 0.95;
constexpr double kMaxResetVerticalSpeed = 0.35;
constexpr double kMaxResetBaseTilt = 0.35;
constexpr double kPolicyStaleTolerance = 0.02;
constexpr double kPolicyFutureTolerance = 0.08;
constexpr double kPolicyResetTimeTolerance = 0.08;

vector_t toEigen(const std::vector<double>& v, std::size_t expectedDim) {
  vector_t out = vector_t::Zero(static_cast<Eigen::Index>(expectedDim));
  const std::size_t n = std::min(expectedDim, v.size());
  for (std::size_t i = 0; i < n; ++i) out[static_cast<Eigen::Index>(i)] = v[i];
  return out;
}

std::vector<double> toDoubleVector(const vector_t& v) {
  std::vector<double> out(static_cast<std::size_t>(v.size()));
  for (Eigen::Index i = 0; i < v.size(); ++i) out[static_cast<std::size_t>(i)] = v[i];
  return out;
}

vector_t interpolateTrajectory(const std::vector<double>& times,
                               const std::vector<vector_t>& traj,
                               double t,
                               std::size_t dim) {
  vector_t result = vector_t::Zero(static_cast<Eigen::Index>(dim));
  if (times.empty() || traj.empty()) return result;
  if (t <= times.front()) return traj.front();
  if (t >= times.back())  return traj.back();

  std::size_t k = 0;
  for (std::size_t i = 0; i + 1 < times.size(); ++i) {
    if (t < times[i + 1]) { k = i; break; }
  }
  const double dt = times[k + 1] - times[k];
  const double alpha = (dt > 1e-12) ? ((t - times[k]) / dt) : 0.0;
  return (1.0 - alpha) * traj[k] + alpha * traj[k + 1];
}

double nominalQ(const std::string& jointName) {
  if (jointName.find("hip_pitch")   != std::string::npos) return -0.20;
  if (jointName.find("knee")        != std::string::npos) return  0.40;
  if (jointName.find("ankle_pitch") != std::string::npos) return -0.20;
  return 0.0;
}

bool isMpcPelvisHeight(const double pelvisZ) {
  return std::isfinite(pelvisZ) &&
         pelvisZ >= kMinMpcPelvisZ &&
         pelvisZ <= kMaxMpcPelvisZ;
}

double verticalVelocityFromState(const vector_t& state, const double /*robotMass*/) {
  if (state.size() <= 2) return 999.0;
  // OCS2 centroidal state uses NORMALIZED centroidal momentum.
  // Linear part is effectively COM velocity, because the original MRT
  // controller sets h_normalized = A(q) * v / robotMass.
  // Therefore state[2] is already vertical COM velocity-like data.
  return state[2];
}

bool isResetState(const vector_t& state, const double robotMass) {
  if (state.size() <= 11) return false;
  const double vz = verticalVelocityFromState(state, robotMass);
  const double pelvisZ = state[8];
  const double pitch = state[10];
  const double roll = state[11];
  return std::isfinite(vz) &&
         std::isfinite(pelvisZ) &&
         std::isfinite(pitch) &&
         std::isfinite(roll) &&
         pelvisZ >= kMinResetPelvisZ &&
         pelvisZ <= kMaxResetPelvisZ &&
         std::abs(vz) <= kMaxResetVerticalSpeed &&
         std::abs(pitch) <= kMaxResetBaseTilt &&
         std::abs(roll) <= kMaxResetBaseTilt;
}

double trackingKp(const std::string& jointName) {
  // G1 gravity at hip ≈52 Nm; at 0.20 rad qDes-qMeas need KP≥260.
  // Higher gains also improve MPC standing stability after blend-in.
  if (jointName.find("hip_pitch")   != std::string::npos) return 200.0;
  if (jointName.find("hip_roll")    != std::string::npos) return 150.0;
  if (jointName.find("hip_yaw")     != std::string::npos) return  40.0;
  if (jointName.find("knee")        != std::string::npos) return  80.0;
  if (jointName.find("ankle_pitch") != std::string::npos) return  60.0;
  if (jointName.find("ankle_roll")  != std::string::npos) return  20.0;
  if (jointName.find("waist_pitch") != std::string::npos) return  65.0;
  if (jointName.find("waist_roll")  != std::string::npos) return  80.0;
  if (jointName.find("waist_yaw")   != std::string::npos) return  25.0;
  if (jointName.find("shoulder")    != std::string::npos) return  10.0;
  if (jointName.find("elbow")       != std::string::npos) return  10.0;
  return 10.0;
}

double trackingKd(const std::string& jointName) {
  if (jointName.find("hip_pitch")   != std::string::npos) return 30.0;
  if (jointName.find("hip_roll")    != std::string::npos) return 20.0;
  if (jointName.find("hip_yaw")     != std::string::npos) return  4.0;
  if (jointName.find("knee")        != std::string::npos) return 10.0;
  if (jointName.find("ankle_pitch") != std::string::npos) return  6.0;
  if (jointName.find("ankle_roll")  != std::string::npos) return  2.0;
  if (jointName.find("waist_pitch") != std::string::npos) return 13.0;
  if (jointName.find("waist_roll")  != std::string::npos) return 16.0;
  if (jointName.find("waist_yaw")   != std::string::npos) return  5.0;
  if (jointName.find("shoulder")    != std::string::npos) return  2.0;
  if (jointName.find("elbow")       != std::string::npos) return  1.5;
  return 1.0;
}

double nominalTorqueLimit(const std::string& jointName) {
  if (jointName.find("hip_")        != std::string::npos) return 88.0;
  if (jointName.find("knee")        != std::string::npos) return 139.0;
  if (jointName.find("ankle")       != std::string::npos) return 50.0;
  if (jointName.find("waist_yaw")   != std::string::npos) return 88.0;
  if (jointName.find("waist_roll")  != std::string::npos) return 50.0;
  if (jointName.find("waist_pitch") != std::string::npos) return 50.0;
  if (jointName.find("shoulder")    != std::string::npos) return 25.0;
  if (jointName.find("elbow")       != std::string::npos) return 25.0;
  if (jointName.find("wrist_roll")  != std::string::npos) return 25.0;
  if (jointName.find("wrist_pitch") != std::string::npos) return  5.0;
  if (jointName.find("wrist_yaw")   != std::string::npos) return  5.0;
  return 50.0;
}

double holdKp(const std::string& jointName) {
  return 0.6 * trackingKp(jointName);
}

double holdKd(const std::string& jointName) {
  return 0.8 * trackingKd(jointName);
}
}  // anonymous namespace

class DrakeCentroidalBridgeNode final : public rclcpp::Node {
public:
  enum class State { WAIT_DRAKE, WAIT_RESET_SERVICE, SEND_RESET, WAIT_POLICY, MPC_TRACKING };

  DrakeCentroidalBridgeNode() : Node("policy_to_command_node") {
    const std::string defaultTaskFile = ament_index_cpp::get_package_share_directory("g1_centroidal_mpc") + "/config/mpc/task.info";
    const std::string defaultUrdfFile = ament_index_cpp::get_package_share_directory("g1_description") + "/urdf/g1_29dof.urdf";
    const std::string defaultReferenceFile = ament_index_cpp::get_package_share_directory("g1_centroidal_mpc") + "/config/command/reference.info";

    declare_parameter<std::string>("task_file", defaultTaskFile);
    declare_parameter<std::string>("urdf_file", defaultUrdfFile);
    declare_parameter<std::string>("reference_file", defaultReferenceFile);
    declare_parameter<double>("control_rate_hz", 500.0);
    declare_parameter<int>("obs_decimation", 5);
    declare_parameter<double>("reset_retry_period_s", 5.0);
    declare_parameter<double>("min_reset_time_s", 0.02);
    declare_parameter<int>("reset_valid_samples", 3);
    declare_parameter<bool>("publish_hold_torques", false);
    declare_parameter<bool>("use_contact_mode_for_observation", false);
    declare_parameter<int>("fixed_observation_mode", 3);
    declare_parameter<double>("mpc_feedforward_scale", kDefaultMpcFeedforwardScale);
    declare_parameter<double>("max_policy_joint_velocity", 6.0);

    taskFile_ = get_parameter("task_file").as_string();
    urdfFile_ = get_parameter("urdf_file").as_string();
    referenceFile_ = get_parameter("reference_file").as_string();
    controlRateHz_ = get_parameter("control_rate_hz").as_double();
    obsDecimation_ = get_parameter("obs_decimation").as_int();
    resetRetryPeriod_ = get_parameter("reset_retry_period_s").as_double();
    minResetTime_ = get_parameter("min_reset_time_s").as_double();
    requiredResetValidSamples_ = std::max(1, static_cast<int>(get_parameter("reset_valid_samples").as_int()));
    publishHoldTorques_ = get_parameter("publish_hold_torques").as_bool();
    useContactModeForObservation_ = get_parameter("use_contact_mode_for_observation").as_bool();
    fixedObservationMode_ = get_parameter("fixed_observation_mode").as_int();
    mpcFeedforwardScale_ = get_parameter("mpc_feedforward_scale").as_double();
    maxPolicyJointVelocity_ = get_parameter("max_policy_joint_velocity").as_double();

    initializeWorkspaceModels();

    auto bestEffort = rclcpp::QoS(1).best_effort();

    jointStateSub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/drake/joint_states", rclcpp::QoS(10),
        std::bind(&DrakeCentroidalBridgeNode::jointStateCallback, this, std::placeholders::_1));

    centroidalStateSub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "/drake/centroidal_state", rclcpp::QoS(10),
        std::bind(&DrakeCentroidalBridgeNode::centroidalStateCallback, this, std::placeholders::_1));

    simTimeSub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "/drake/sim_time", rclcpp::QoS(10),
        [this](std_msgs::msg::Float64MultiArray::SharedPtr m) {
          if (!m->data.empty()) {
            std::lock_guard<std::mutex> lock(mutex_);
            const double newT = m->data[0];
            // Detect Drake respawn: sim time jumped backward → stale MPC policy is invalid.
            if (newT < simTime_ - 1.0) {
              RCLCPP_WARN(get_logger(),
                  "Drake sim time reset (%.3f→%.3f): returning to HOLD, discarding policy",
                  simTime_, newT);
              state_ = State::WAIT_RESET_SERVICE;
              clearPolicy();
              commandStartupPin(true);
              resetSentTime_ = newT;
              resetValidSampleCount_ = 0;
              resetObservationDelayTicks_ = 0;
            }
            simTime_ = newT;
          }
        });

    footContactsSub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "/drake/foot_contacts", rclcpp::QoS(10),
        std::bind(&DrakeCentroidalBridgeNode::footContactsCallback, this, std::placeholders::_1));

    policySub_ = create_subscription<MpcPolicy>(
        "/g1/mpc_policy", bestEffort,
        std::bind(&DrakeCentroidalBridgeNode::policyCallback, this, std::placeholders::_1));

    observationPub_ = create_publisher<MpcObsMsg>("/g1/mpc_observation", bestEffort);
    torquePub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/drake/joint_torques", rclcpp::QoS(10));
    startupPinPub_ = create_publisher<std_msgs::msg::Bool>("/drake/startup_pin", rclcpp::QoS(10));
    resetClient_ = create_client<ResetSrv>("/g1/mpc_reset");

    const auto periodUs = static_cast<int>(1e6 / controlRateHz_);
    controlTimer_ = create_wall_timer(
        std::chrono::microseconds(periodUs),
        std::bind(&DrakeCentroidalBridgeNode::controlLoop, this));

    RCLCPP_INFO(get_logger(),
                "DrakeCentroidalBridgeNode ready | stateDim=%zu inputDim=%zu joints=%zu rate=%.0fHz mode=%s ff=%.2f vel_clip=%.2f",
                stateDim_, inputDim_, jointDim_, controlRateHz_,
                useContactModeForObservation_ ? "contact" : "fixed",
                mpcFeedforwardScale_, maxPolicyJointVelocity_);
  }

private:
  void initializeWorkspaceModels() {
    // Build model stack exactly as CentroidalMpcInterface does.
    // Deliberately avoids CentroidalMpcRobotModel and DynamicsHelperFunctions
    // because both transitively include MpcRobotModelBase.h which pulls in
    // boost/property_tree/info_parser.hpp → boost/multi_index, breaking compilation
    // when humanoid_centroidal_mpc injects -DBOOST_MPL_LIMIT_LIST_SIZE=30.
    modelSettings_ = std::make_unique<ModelSettings>(taskFile_, urdfFile_, "centroidal_mpc_", false);

    pinocchioInterface_ = std::make_unique<PinocchioInterface>(
        createCustomPinocchioInterface(taskFile_, urdfFile_, *modelSettings_));

    centroidalModelInfo_ = centroidal_model::createCentroidalModelInfo(
        *pinocchioInterface_,
        centroidal_model::loadCentroidalType(taskFile_),
        centroidal_model::loadDefaultJointState(pinocchioInterface_->getModel().nq - 6, referenceFile_),
        modelSettings_->contactNames3DoF,
        modelSettings_->contactNames6DoF);

    jointNamesMpc_ = modelSettings_->mpcModelJointNames;
    jointDim_ = static_cast<std::size_t>(centroidalModelInfo_.actuatedDofNum);
    stateDim_ = static_cast<std::size_t>(centroidalModelInfo_.stateDim);
    inputDim_ = static_cast<std::size_t>(centroidalModelInfo_.inputDim);

    // Foot contact frame indices for Pinocchio inverse dynamics
    const auto& pinModel = pinocchioInterface_->getModel();
    leftFootFrame_  = pinModel.getFrameId("foot_l_contact");
    rightFootFrame_ = pinModel.getFrameId("foot_r_contact");
    RCLCPP_INFO(rclcpp::get_logger("init"),
        "Pinocchio: nq=%d nv=%d  L_frame=%lu R_frame=%lu",
        pinModel.nq, pinModel.nv, leftFootFrame_, rightFootFrame_);

    qNominal_ = vector_t::Zero(static_cast<Eigen::Index>(jointDim_));
    qMeasured_ = vector_t::Zero(static_cast<Eigen::Index>(jointDim_));
    vMeasured_ = vector_t::Zero(static_cast<Eigen::Index>(jointDim_));
    centroidalState12_ = vector_t::Zero(12);

    for (std::size_t i = 0; i < jointNamesMpc_.size() && i < jointDim_; ++i) {
      qNominal_[static_cast<Eigen::Index>(i)] = nominalQ(jointNamesMpc_[i]);
    }

    currentObservation_.state = vector_t::Zero(static_cast<Eigen::Index>(stateDim_));
    currentObservation_.input = vector_t::Zero(static_cast<Eigen::Index>(inputDim_));
  }

  // Pinocchio inverse dynamics, matching humanoid_common_mpc::computeJointTorques().
  // The important part is the floating-base acceleration solve; using only
  // nle.tail() - J^T*w is not equivalent for a floating-base humanoid.
  vector_t computeBaseAccelerationLocal(const matrix_t& M,
                                        const vector_t& nle,
                                        const vector_t& qddJoints,
                                        const vector_t& externalForcesInJointSpace) const {
    matrix3_t MbbLin = M.topLeftCorner(3, 3);
    matrix3_t MbbAng = M.block(3, 3, 3, 3);
    matrix_t Mbj = M.block(0, 6, 6, qddJoints.size());

    vector6_t intermediate = -nle.head(6) - Mbj * qddJoints + externalForcesInJointSpace.head(6);

    vector6_t baseAcc;
    baseAcc.head(3) = MbbLin.inverse() * intermediate.head(3);
    baseAcc.tail(3) = MbbAng.inverse() * intermediate.tail(3);
    return baseAcc;
  }

  vector_t computeJointTorquesRaw(
      const vector_t& q, const vector_t& qd,
      const Eigen::Matrix<double,6,1>& wl, const Eigen::Matrix<double,6,1>& wr) const
  {
    auto& model = pinocchioInterface_->getModel();
    auto& data  = pinocchioInterface_->getData();
    const int nv = model.nv;
    const int nj = static_cast<int>(jointDim_);

    if (q.size() != nv || qd.size() != nv) {
      throw std::runtime_error("Pinocchio q/qd dimension mismatch");
    }

    pinocchio::crba(model, data, q);
    data.M.triangularView<Eigen::StrictlyLower>() =
        data.M.transpose().triangularView<Eigen::StrictlyLower>();
    pinocchio::computeAllTerms(model, data, q, qd);  // populates data.nle, data.M

    matrix_t Jl = matrix_t::Zero(6, nv);
    matrix_t Jr = matrix_t::Zero(6, nv);
    pinocchio::computeFrameJacobian(model, data, q, leftFootFrame_,
        pinocchio::LOCAL_WORLD_ALIGNED, Jl);
    pinocchio::computeFrameJacobian(model, data, q, rightFootFrame_,
        pinocchio::LOCAL_WORLD_ALIGNED, Jr);

    const vector_t external = Jl.transpose() * wl + Jr.transpose() * wr;
    const vector_t qddJoints = vector_t::Zero(nj);
    const vector6_t baseAcc = computeBaseAccelerationLocal(data.M, data.nle, qddJoints, external);

    vector_t qdd(nv);
    qdd << baseAcc, qddJoints;

    return data.M.bottomRows(nj) * qdd + data.nle.tail(nj) - external.tail(nj);
  }

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!mappingBuilt_ && !msg->name.empty()) {
      nDrakeActuators_ = static_cast<int>(msg->name.size());
      drakeActuatorNames_ = msg->name;
      mpcToDrake_.assign(jointDim_, -1);
      int matched = 0;
      for (std::size_t mi = 0; mi < jointDim_; ++mi) {
        for (int di = 0; di < nDrakeActuators_; ++di) {
          if (jointNamesMpc_[mi] == msg->name[static_cast<std::size_t>(di)]) {
            mpcToDrake_[mi] = di;
            ++matched;
            break;
          }
        }
      }
      mappingBuilt_ = true;
      RCLCPP_INFO(get_logger(), "Joint map built: %d/%zu MPC joints matched in %d Drake actuators",
                  matched, jointDim_, nDrakeActuators_);
    }
    if (!mappingBuilt_) return;
    for (std::size_t mi = 0; mi < jointDim_; ++mi) {
      const int di = mpcToDrake_[mi];
      if (di < 0) continue;
      if (di < static_cast<int>(msg->position.size())) {
        qMeasured_[static_cast<Eigen::Index>(mi)] = msg->position[static_cast<std::size_t>(di)];
      }
      if (di < static_cast<int>(msg->velocity.size())) {
        vMeasured_[static_cast<Eigen::Index>(mi)] = msg->velocity[static_cast<std::size_t>(di)];
      }
    }
    jointsReceived_ = true;
  }

  void centroidalStateCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (msg->data.size() < 12) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < 12; ++i) centroidalState12_[i] = msg->data[static_cast<std::size_t>(i)];

    // Validate normalized centroidal momentum units: h_lin ~= COM velocity, h_ang = angular momentum / mass.
    const double h_lin_mag = centroidalState12_.segment<3>(0).norm();
    const double h_ang_mag = centroidalState12_.segment<3>(3).norm();
    const double pelvis_z  = centroidalState12_[8];
    if (h_lin_mag > 20.0)
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "[state] normalized h_lin/COM velocity=%.3f is unexpectedly large", h_lin_mag);
    if (h_ang_mag > 50.0)
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "[state] normalized h_ang=%.3f is unexpectedly large", h_ang_mag);
    if (pelvis_z > 2.0 || (drakeStateValid_ && pelvis_z < 0.05))
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "[state] pelvis_z=%.3f out of range [0.05, 2.0]", pelvis_z);

    drakeStateValid_ = true;
  }

  void footContactsCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (msg->data.size() < 4) return;
    std::lock_guard<std::mutex> lock(mutex_);
    footContactsReceived_ = true;
    leftContact_ = (msg->data[0] > 0.5) || (msg->data[2] > kContactForceThreshold);
    rightContact_ = (msg->data[1] > 0.5) || (msg->data[3] > kContactForceThreshold);
  }

  void clearPolicy() {
    policyReceived_ = false;
    firstPolicyLogged_ = false;
    policyTimes_.clear();
    policyStates_.clear();
    policyInputs_.clear();
    firstPolicyTime_ = simTime_;
  }

  void policyCallback(const MpcPolicy::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (msg->time_trajectory.empty() ||
        msg->state_trajectory.empty() ||
        msg->input_trajectory.empty()) {
      return;
    }
    if (state_ != State::WAIT_POLICY && state_ != State::MPC_TRACKING) {
      return;
    }
    const double policyStart = static_cast<double>(msg->time_trajectory.front());
    const double policyEnd = static_cast<double>(msg->time_trajectory.back());
    if (policyEnd + kPolicyStaleTolerance < simTime_ ||
        policyStart > simTime_ + kPolicyFutureTolerance) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "Dropping stale MPC policy: sim_t=%.3f policy=[%.3f, %.3f]",
          simTime_, policyStart, policyEnd);
      return;
    }
    if (state_ == State::WAIT_POLICY &&
        std::abs(policyStart - resetSentTime_) > kPolicyResetTimeTolerance) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "Dropping policy from a different reset: reset_t=%.3f policy_t0=%.3f sim_t=%.3f",
          resetSentTime_, policyStart, simTime_);
      return;
    }
    policyTimes_.clear();
    policyStates_.clear();
    policyInputs_.clear();
    for (const auto t : msg->time_trajectory) policyTimes_.push_back(static_cast<double>(t));
    for (const auto& s : msg->state_trajectory)
      policyStates_.push_back(toEigen(s.value, stateDim_));
    for (const auto& u : msg->input_trajectory)
      policyInputs_.push_back(toEigen(u.value, inputDim_));
    if (!policyReceived_) {
      firstPolicyTime_ = simTime_;  // record time of first policy
    }
    policyReceived_ = true;
    if (!firstPolicyLogged_) {
      firstPolicyLogged_ = true;
      RCLCPP_INFO(get_logger(),
                  "First MPC policy received | pts=%zu t0=%.4f tf=%.4f",
                  policyTimes_.size(),
                  policyTimes_.front(),
                  policyTimes_.back());
    }
  }

  void updateCurrentObservation() {
    currentObservation_.state.setZero();
    currentObservation_.input.setZero();
    currentObservation_.state.segment(0, 12) = centroidalState12_;
    currentObservation_.state.tail(static_cast<Eigen::Index>(jointDim_)) = qMeasured_;

    // Dimension sanity check.
    if (currentObservation_.state.size() != static_cast<Eigen::Index>(stateDim_)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
          "[obs] state dim mismatch: got %ld expected %zu — zeroing",
          currentObservation_.state.size(), stateDim_);
      currentObservation_.state.setZero();
      return;
    }

    if (useContactModeForObservation_) {
      const contact_flag_t contactFlags = currentContactFlags();
      currentObservation_.mode = stanceLeg2ModeNumber(contactFlags);
    } else {
      currentObservation_.mode = static_cast<size_t>(std::max(0, fixedObservationMode_));
    }
    currentObservation_.time = simTime_;
  }

  contact_flag_t currentContactFlags() const {
    if (footContactsReceived_) {
      // NEW FALLBACK: if both feet reported off but pelvis is well above ground, force double support
      if (!leftContact_ && !rightContact_ && centroidalState12_.size() > 8) {
        const double pelvisZ = centroidalState12_[8];
        if (pelvisZ > 0.65) {   // your proposed threshold
          return {true, true};
        }
      }
      return {leftContact_, rightContact_};
    }
    return {true, true};
  }

  void publishObservation() {
    if (obsDecimation_ > 1 && (tickCount_ % obsDecimation_) != 0) return;
    updateCurrentObservation();

    // Don't publish physically impossible observations to MPC.
    // pelvis_z == 0 means Drake failed and the sanity clamp zeroed it.
    // Publishing this would cause the MPC's foot constraint to become infeasible → crash.
    const double pelvisZ = (centroidalState12_.size() >= 9) ? centroidalState12_[8] : 0.5;
    if (pelvisZ < 0.05 || pelvisZ > 2.0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "Suppressing observation: pelvis_z=%.3f out of valid range", pelvisZ);
      return;
    }

    MpcObsMsg msg;
    msg.time = static_cast<float>(currentObservation_.time);
    msg.mode = static_cast<uint64_t>(currentObservation_.mode);
    msg.state.value = toDoubleVector(currentObservation_.state);
    msg.input.value.assign(inputDim_, 0.0);
    observationPub_->publish(msg);
  }

  bool sendReset() {
    updateCurrentObservation();

    if (simTime_ < minResetTime_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "Waiting to reset MPC: sim_t=%.3f < min_reset_time_s=%.3f",
          simTime_, minResetTime_);
      return false;
    }

    // ── Guard: don't reset MPC if robot is in an unphysical state ──
    const Eigen::Index pelvis_z_idx = 8;  // base z in centroidal state (12-element)
    const double pelvisZ = (currentObservation_.state.size() > pelvis_z_idx)
        ? currentObservation_.state[pelvis_z_idx]
        : -999.0;
    const double vz = verticalVelocityFromState(
        currentObservation_.state, centroidalModelInfo_.robotMass);
    if (currentObservation_.state.size() <= pelvis_z_idx ||
        !isResetState(currentObservation_.state, centroidalModelInfo_.robotMass)) {
      // Pelvis height too low – robot likely falling; skip reset and stay in hold.
      resetValidSampleCount_ = 0;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Skipping MPC reset: pelvis_z=%.3f vz=%.3f pitch=%.3f roll=%.3f is invalid.",
                           pelvisZ,
                           vz,
                           currentObservation_.state.size() > 10 ? currentObservation_.state[10] : 999.0,
                           currentObservation_.state.size() > 11 ? currentObservation_.state[11] : 999.0);
      return false;
    }

    ++resetValidSampleCount_;
    if (resetValidSampleCount_ < requiredResetValidSamples_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
          "Waiting for reset-stable samples: %d/%d pelvis_z=%.3f vz=%.3f",
          resetValidSampleCount_, requiredResetValidSamples_, pelvisZ, vz);
      return false;
    }

    clearPolicy();
    commandStartupPin(true);

    vector_t targetState = currentObservation_.state;
    targetState.head(6).setZero();
    if (targetState.size() >= 12) {
      targetState[10] = 0.0;  // pitch
      targetState[11] = 0.0;  // roll
    }

    auto req = std::make_shared<ResetSrv::Request>();
    req->reset = true;

    ocs2_ros2_msgs::msg::MpcTargetTrajectories tgt;
    tgt.time_trajectory = {static_cast<float>(currentObservation_.time)};

    ocs2_ros2_msgs::msg::MpcState st;
    st.value = toDoubleVector(targetState);
    tgt.state_trajectory = {st};

    ocs2_ros2_msgs::msg::MpcInput in;
    in.value.assign(inputDim_, 0.0);
    tgt.input_trajectory = {in};

    req->target_trajectories = tgt;
    resetClient_->async_send_request(req);

    resetSentTime_ = simTime_;
    resetValidSampleCount_ = 0;
    resetObservationDelayTicks_ =
        static_cast<int>(std::ceil(0.05 * std::max(1.0, controlRateHz_)));
    RCLCPP_INFO(get_logger(), "MPC reset sent at t=%.3f", simTime_);
    return true;
  }

  void commandStartupPin(const bool enable, const bool force = false) {
    if (!startupPinPub_) return;
    if (!force && startupPinCommanded_ == enable) return;
    startupPinCommanded_ = enable;

    std_msgs::msg::Bool msg;
    msg.data = enable;
    startupPinPub_->publish(msg);
    RCLCPP_INFO(get_logger(), "Drake startup pin command: %s", enable ? "ON" : "OFF");
  }

  void applyHoldTorques() {
    updateCurrentObservation();

    // Weight-compensating wrenches: split robot weight equally between stance feet
    const contact_flag_t contactFlags = currentContactFlags();
    const double nLegs = (contactFlags[0] ? 1.0 : 0.0) + (contactFlags[1] ? 1.0 : 0.0);
    const double fz = (nLegs > 0)
        ? (centroidalModelInfo_.robotMass * kGravity / nLegs)
        : (centroidalModelInfo_.robotMass * kGravity * 0.5);

    Eigen::Matrix<double,6,1> wl = Eigen::Matrix<double,6,1>::Zero();
    Eigen::Matrix<double,6,1> wr = Eigen::Matrix<double,6,1>::Zero();
    if (contactFlags[0] || nLegs == 0) wl[2] = fz;
    if (contactFlags[1] || nLegs == 0) wr[2] = fz;

    // Build generalized coords from centroidal observation.
    // q = [base_pos(3), euler_ZYX(3), joint_angles(nj)] — matches SphericalZYX composite joint.
    const int nv = pinocchioInterface_->getModel().nv;  // 6 (floating) + nJoints
    vector_t q  = centroidal_model::getGeneralizedCoordinates(
        currentObservation_.state, centroidalModelInfo_);
    vector_t qd = vector_t::Zero(nv);
    // Floating-base twist remains zero because centroidalState12 stores momentum,
    // not base twist. Joint rates are still fed to inverse dynamics below.
    qd.tail(static_cast<Eigen::Index>(jointDim_)) = vMeasured_;

    vector_t tauFf = vector_t::Zero(static_cast<Eigen::Index>(jointDim_));
    try {
      tauFf = computeJointTorquesRaw(q, qd, wl, wr);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Hold IDyn failed: %s", e.what());
    }

    vector_t tau = kHoldFeedforwardScale * tauFf;
    for (std::size_t i = 0; i < jointDim_ && i < jointNamesMpc_.size(); ++i) {
      const auto idx = static_cast<Eigen::Index>(i);
      tau[idx] += holdKp(jointNamesMpc_[i]) * (qNominal_[idx] - qMeasured_[idx]) +
                  holdKd(jointNamesMpc_[i]) * (-vMeasured_[idx]);
    }
    publishTorques(tau);
  }

  void applyMpcTorques() {
    if (!policyReceived_) {
      applyHoldTorques();
      return;
    }

    updateCurrentObservation();

    const double evalTime = currentObservation_.time + kPolicyLookahead;
    const vector_t policyState = interpolateTrajectory(policyTimes_, policyStates_, evalTime, stateDim_);
    const vector_t policyInput = interpolateTrajectory(policyTimes_, policyInputs_, evalTime, inputDim_);

    // Extract desired joint motion using OCS2 free functions (no MpcRobotModelBase needed)
    const vector_t qDes  = centroidal_model::getJointAngles(policyState, centroidalModelInfo_);
    vector_t qdDes = centroidal_model::getJointVelocities(policyInput, centroidalModelInfo_);
    if (maxPolicyJointVelocity_ > 0.0) {
      qdDes =
          qdDes.cwiseMax(-maxPolicyJointVelocity_).cwiseMin(maxPolicyJointVelocity_);
    }

    // Diagnostic: print desired vs measured pelvis height and knee angles every 1 second (500 ticks at 500 Hz)
    static int debug_counter = 0;
    if (debug_counter++ % 500 == 0) {
      // Pelvis Z is at index 8 in the 35‑dimensional state (12 centroidal + 23 joints)
      double pelvis_des = policyState.size() > 8 ? policyState[8] : 0.0;
      double pelvis_meas = currentObservation_.state.size() > 8 ? currentObservation_.state[8] : 0.0;

      // Find index of left knee joint (example: "left_knee_joint" is usually at position ~3 in joint order)
      // You can either hardcode the index after inspecting your joint order, or search by name.
      int knee_idx = -1;
      for (size_t i = 0; i < jointNamesMpc_.size(); ++i) {
          if (jointNamesMpc_[i].find("knee") != std::string::npos) {
              knee_idx = static_cast<int>(i);
              break;
          }
      }
      double knee_des = (knee_idx >= 0 && qDes.size() > knee_idx) ? qDes[knee_idx] : 0.0;
      double knee_meas = (knee_idx >= 0 && qMeasured_.size() > knee_idx) ? qMeasured_[knee_idx] : 0.0;

      RCLCPP_INFO(get_logger(),
                  "MPC debug: pelvis_z_des=%.3f meas=%.3f, knee_angle_des=%.3f meas=%.3f",
                  pelvis_des, pelvis_meas, knee_des, knee_meas);
    }

    // Contact wrenches from policy input: [force(3), torque(3)] per contact
    Eigen::Matrix<double,6,1> wl = Eigen::Matrix<double,6,1>::Zero();
    Eigen::Matrix<double,6,1> wr = Eigen::Matrix<double,6,1>::Zero();
    wl.head<3>() = centroidal_model::getContactForces(policyInput, 0, centroidalModelInfo_);
    wl.tail<3>() = centroidal_model::getContactTorques(policyInput, 0, centroidalModelInfo_);
    wr.head<3>() = centroidal_model::getContactForces(policyInput, 1, centroidalModelInfo_);
    wr.tail<3>() = centroidal_model::getContactTorques(policyInput, 1, centroidalModelInfo_);

    // Build generalized coords from current observation.
    const int nv = pinocchioInterface_->getModel().nv;
    vector_t q  = centroidal_model::getGeneralizedCoordinates(
        currentObservation_.state, centroidalModelInfo_);

    // ***** CRITICAL FIX: Zero floating-base velocity to avoid NaN torques *****
    // Using centroidal momentum as base angular velocity is physically incorrect
    // and causes RNEA to diverge. Setting base twist = 0 is stable and sufficient
    // for feedforward + PD tracking.
    vector_t qd = vector_t::Zero(nv);
    qd.tail(static_cast<Eigen::Index>(jointDim_)) = vMeasured_;   // only joint velocities

    vector_t tauFf = vector_t::Zero(static_cast<Eigen::Index>(jointDim_));
    try {
      tauFf = computeJointTorquesRaw(q, qd, wl, wr);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "MPC IDyn failed: %s", e.what());
    }

    // MPC tracking torques (PD + feedforward)
    vector_t tau_mpc = mpcFeedforwardScale_ * tauFf;
    for (std::size_t i = 0; i < jointDim_ && i < jointNamesMpc_.size(); ++i) {
      const auto idx = static_cast<Eigen::Index>(i);
      tau_mpc[idx] += trackingKp(jointNamesMpc_[i]) * (qDes[idx] - qMeasured_[idx]) +
                      trackingKd(jointNamesMpc_[i]) * (qdDes[idx] - vMeasured_[idx]);
    }

    // Hold torques — computed using the MPC contact wrenches but nominal joint targets.
    // This keeps the robot supported while the MPC policy is still ramping in.
    const contact_flag_t holdContactFlags = currentContactFlags();
    const double hnLegs = (holdContactFlags[0] ? 1.0 : 0.0) + (holdContactFlags[1] ? 1.0 : 0.0);
    const double hfz = (hnLegs > 0)
        ? (centroidalModelInfo_.robotMass * kGravity / hnLegs)
        : (centroidalModelInfo_.robotMass * kGravity * 0.5);
    Eigen::Matrix<double,6,1> hwl = Eigen::Matrix<double,6,1>::Zero();
    Eigen::Matrix<double,6,1> hwr = Eigen::Matrix<double,6,1>::Zero();
    if (holdContactFlags[0] || hnLegs == 0) hwl[2] = hfz;
    if (holdContactFlags[1] || hnLegs == 0) hwr[2] = hfz;
    vector_t htauFf = vector_t::Zero(static_cast<Eigen::Index>(jointDim_));
    try { htauFf = computeJointTorquesRaw(q, qd, hwl, hwr); } catch (...) {}

    vector_t tau_hold = kHoldFeedforwardScale * htauFf;
    for (std::size_t i = 0; i < jointDim_ && i < jointNamesMpc_.size(); ++i) {
      const auto idx = static_cast<Eigen::Index>(i);
      tau_hold[idx] += holdKp(jointNamesMpc_[i]) * (qNominal_[idx] - qMeasured_[idx]) +
                       holdKd(jointNamesMpc_[i]) * (-vMeasured_[idx]);
    }

    // Blend: start fully in hold, ramp to MPC over 1.5 s.
    // This prevents the robot losing gravity support during the stand-up transition.
    const double dt_since_first_policy = currentObservation_.time - firstPolicyTime_;
    const double alpha = std::min(1.0, dt_since_first_policy / 1.5);
    vector_t tau = (1.0 - alpha) * tau_hold + alpha * tau_mpc;

    publishTorques(tau);
    commandStartupPin(false);
  }

  void publishTorques(const vector_t& tauMpc) {
    if (!mappingBuilt_ || nDrakeActuators_ <= 0) return;

    Eigen::VectorXd tauDrake = Eigen::VectorXd::Zero(nDrakeActuators_);
    for (std::size_t mi = 0; mi < jointDim_; ++mi) {
      const int di = mpcToDrake_[mi];
      if (di >= 0 && di < nDrakeActuators_) {
        tauDrake[di] = tauMpc[static_cast<Eigen::Index>(mi)];
      }
    }

    for (int i = 0; i < nDrakeActuators_; ++i) {
      if (!std::isfinite(tauDrake[i])) tauDrake[i] = 0.0;
      double limit = kGlobalTauLimit;
      if (i < static_cast<int>(drakeActuatorNames_.size())) {
        limit = std::min(kGlobalTauLimit, nominalTorqueLimit(drakeActuatorNames_[i]));
      }
      tauDrake[i] = std::clamp(tauDrake[i], -limit, limit);
    }

    std_msgs::msg::Float64MultiArray msg;
    msg.data.assign(tauDrake.data(), tauDrake.data() + tauDrake.size());
    torquePub_->publish(msg);
  }

  void publishZeroTorques() {
    if (nDrakeActuators_ <= 0) return;
    std_msgs::msg::Float64MultiArray msg;
    msg.data.assign(static_cast<std::size_t>(nDrakeActuators_), 0.0);
    torquePub_->publish(msg);
  }

  void controlLoop() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++tickCount_;

    switch (state_) {
      case State::WAIT_DRAKE:
        if (drakeStateValid_ && jointsReceived_ && mappingBuilt_) {
          state_ = State::WAIT_RESET_SERVICE;
        }
        return;

      case State::WAIT_RESET_SERVICE:
        // Also enforce 2s minimum after startup pin was engaged (prevents rapid
        // re-reset while MPC SQP warm-start is still inconsistent).
        if (resetClient_->service_is_ready() &&
            simTime_ >= minResetTime_ &&
            (simTime_ - pinOnSimTime_) >= 2.0) {
          state_ = State::SEND_RESET;
        }
        if (publishHoldTorques_) applyHoldTorques();
        return;

      case State::SEND_RESET:
        if (sendReset()) {
          state_ = State::WAIT_POLICY;
        } else {
          state_ = State::WAIT_RESET_SERVICE;
          if (publishHoldTorques_) applyHoldTorques();
        }
        return;

      case State::WAIT_POLICY:
        if (resetObservationDelayTicks_ > 0) {
          --resetObservationDelayTicks_;
        } else {
          publishObservation();
        }
        if (publishHoldTorques_) applyHoldTorques();
        if (policyReceived_) {
          state_ = State::MPC_TRACKING;
          RCLCPP_INFO(get_logger(), "Switching to MPC tracking");
        } else if ((simTime_ - resetSentTime_) > resetRetryPeriod_) {
          state_ = State::SEND_RESET;
          RCLCPP_WARN(get_logger(), "No policy after reset timeout; retrying reset");
        }
        break;

      case State::MPC_TRACKING:
        publishObservation();
        updateCurrentObservation();
        if (!isMpcPelvisHeight(currentObservation_.state[8])) {
          const double pelvisZ = currentObservation_.state[8];
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
              "Leaving MPC tracking: pelvis_z=%.3f is unsafe.", pelvisZ);
          commandStartupPin(true);
          clearPolicy();
          publishZeroTorques();
          resetValidSampleCount_ = 0;
          resetObservationDelayTicks_ = 0;
          pinOnSimTime_ = simTime_;   // enforce 2s hold before next MPC reset
          state_ = State::WAIT_RESET_SERVICE;
          if (publishHoldTorques_) applyHoldTorques();
          break;
        }
        applyMpcTorques();
        break;
    }

    if ((tickCount_ % static_cast<int>(std::max(1.0, controlRateHz_))) == 0) {
      const double pelvisZ = (centroidalState12_.size() >= 9) ? centroidalState12_[8] : 0.0;
      const char* phase = (state_ == State::MPC_TRACKING)
          ? (!isMpcPelvisHeight(pelvisZ) ? "MPC(unsafe)" : (policyReceived_ ? "MPC" : "MPC(wait)"))
          : "HOLD";
      RCLCPP_INFO(get_logger(), "[bridge] t=%.2f phase=%s pelvis_z=%.3f", simTime_, phase, pelvisZ);
    }
  }

  // ── Member variables ─────────────────────────────────────────────────────
  std::string taskFile_;
  std::string urdfFile_;
  std::string referenceFile_;

  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  CentroidalModelInfo centroidalModelInfo_;
  pinocchio::FrameIndex leftFootFrame_{0}, rightFootFrame_{0};

  std::size_t stateDim_{0};
  std::size_t inputDim_{0};
  std::size_t jointDim_{0};

  std::vector<std::string> jointNamesMpc_;
  std::vector<std::string> drakeActuatorNames_;
  std::vector<int> mpcToDrake_;
  int nDrakeActuators_{0};

  vector_t qNominal_;
  vector_t qMeasured_;
  vector_t vMeasured_;
  vector_t centroidalState12_;

  SystemObservation currentObservation_;

  bool mappingBuilt_{false};
  bool jointsReceived_{false};
  bool drakeStateValid_{false};
  bool footContactsReceived_{false};
  bool policyReceived_{false};
  bool firstPolicyLogged_{false};
  bool startupPinCommanded_{true};

  bool leftContact_{true};
  bool rightContact_{true};

  double simTime_{0.0};
  double resetSentTime_{0.0};
  double pinOnSimTime_{-100.0};  // sim time of last pin-ON; -100 = never
  double firstPolicyTime_{0.0};
  double controlRateHz_{500.0};
  double resetRetryPeriod_{5.0};
  double minResetTime_{0.02};
  double mpcFeedforwardScale_{kDefaultMpcFeedforwardScale};
  double maxPolicyJointVelocity_{6.0};
  bool publishHoldTorques_{false};
  bool useContactModeForObservation_{false};
  int fixedObservationMode_{3};
  int requiredResetValidSamples_{3};
  int resetValidSampleCount_{0};
  int resetObservationDelayTicks_{0};
  int obsDecimation_{5};
  int tickCount_{0};

  State state_{State::WAIT_DRAKE};

  std::vector<double> policyTimes_;
  std::vector<vector_t> policyStates_;
  std::vector<vector_t> policyInputs_;

  std::mutex mutex_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr jointStateSub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr centroidalStateSub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr simTimeSub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr footContactsSub_;
  rclcpp::Subscription<MpcPolicy>::SharedPtr policySub_;

  rclcpp::Publisher<MpcObsMsg>::SharedPtr observationPub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr torquePub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr startupPinPub_;
  rclcpp::Client<ResetSrv>::SharedPtr resetClient_;
  rclcpp::TimerBase::SharedPtr controlTimer_;
};

}  // namespace ocs2::humanoid

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ocs2::humanoid::DrakeCentroidalBridgeNode>());
  rclcpp::shutdown();
  return 0;
}