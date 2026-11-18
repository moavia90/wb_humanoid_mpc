#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

// OCS2 ROS 2 interfaces
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_ros2_interfaces/mrt/MRT_ROS_Interface.h>
#include <ocs2_ros2_interfaces/common/RosMsgConversions.h>
#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

// Centroidal MPC and Pinocchio models
#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h>
#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>
#include <humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h>
#include <humanoid_common_mpc_ros2/visualization/HumanoidVisualizer.h>
#include <humanoid_mpc_msgs/msg/walking_velocity_command.hpp>

// Drake simulation interface and robot model
#include <humanoid_drake_sim/DrakeSimInterface.h>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotState.h>
#include <robot_model/RobotJointAction.h>

using namespace ocs2;
using namespace ocs2::humanoid;

namespace {

robot::model::RobotState makeInitialState(
    const std::string& urdfFile, CentroidalMpcInterface& interface) {
  robot::model::RobotDescription description(urdfFile);
  robot::model::RobotState state(description, 2);
  state.setConfigurationToZero();

  const vector_t& initialMpcState = interface.getInitialState();
  const auto& mpcModel = interface.getMpcRobotModel();
  vector3_t basePosition = mpcModel.getBasePosition(initialMpcState);
  // Calibrate initial floating base spawn height to ensure foot contact at z=0.0 m
  // with static equilibrium normal forces supporting robot mass (z_spawn = 0.791068 m).
  // In task.info nominal posture (-0.05, 0.10, -0.05), foot bottom is at +0.001432 m.
  constexpr double kStanceHeightGroundingOffset = 0.001432;
  basePosition.z() -= kStanceHeightGroundingOffset;
  state.setRootPositionInWorldFrame(basePosition);

  const vector_t jointAngles = mpcModel.getJointAngles(initialMpcState);
  const auto jointIndices = description.getJointIndices(
      interface.modelSettings().mpcModelJointNames);
  if (jointIndices.size() != static_cast<size_t>(jointAngles.size())) {
    throw std::runtime_error(
        "Initial MPC joint vector does not match configured joint names");
  }
  for (size_t i = 0; i < jointIndices.size(); ++i) {
    state.setJointPosition(jointIndices[i],
                           jointAngles[static_cast<Eigen::Index>(i)]);
  }
  return state;
}

struct ContinuousYawUnwrapper {
  bool initialized = false;
  double continuousYaw = 0.0;

  void reset() {
    initialized = false;
    continuousYaw = 0.0;
  }

  double unwrap(double rawYaw) {
    if (!std::isfinite(rawYaw)) {
      return continuousYaw;
    }
    if (!initialized) {
      continuousYaw = rawYaw;
      initialized = true;
    } else {
      continuousYaw += std::remainder(rawYaw - continuousYaw, 2.0 * M_PI);
    }
    return continuousYaw;
  }
};

void updateMpcState(vector_t& mpcState,
                    const robot::model::RobotState& robotState,
                    const CentroidalMpcRobotModel<scalar_t>& mpcRobotModel,
                    PinocchioInterface& pinocchioInterface,
                    const std::vector<size_t>& mpcJointIndices,
                    ContinuousYawUnwrapper& yawUnwrapper) {
  const auto& info = mpcRobotModel.getCentroidalModelInfo();
  vector3_t euler_zyx =
      quaternionToEulerZYX(robotState.getRootRotationLocalToWorldFrame());
  euler_zyx(0) = yawUnwrapper.unwrap(euler_zyx(0));

  vector_t qPinocchio(info.generalizedCoordinatesNum);
  qPinocchio.head<3>() = robotState.getRootPositionInWorldFrame();
  qPinocchio.segment<3>(3) = euler_zyx;
  qPinocchio.tail(mpcRobotModel.getJointDim()) =
      robotState.getJointPositions(mpcJointIndices);

  vector_t vPinocchio(info.generalizedCoordinatesNum);
  vPinocchio.head<3>() =
      robotState.getRootRotationLocalToWorldFrame() *
      robotState.getRootLinearVelocityInLocalFrame();
  vPinocchio.segment<3>(3) =
      getEulerAnglesZyxDerivativesFromLocalAngularVelocity<scalar_t>(
          euler_zyx, robotState.getRootAngularVelocityInLocalFrame());
  vPinocchio.tail(mpcRobotModel.getJointDim()) =
      robotState.getJointVelocities(mpcJointIndices);

  updateCentroidalDynamics(pinocchioInterface, info, qPinocchio);
  const auto& A = getCentroidalMomentumMatrix(pinocchioInterface);

  centroidal_model::getNormalizedMomentum(mpcState, info).noalias() =
      A * vPinocchio / info.robotMass;
  centroidal_model::getGeneralizedCoordinates(mpcState, info) = qPinocchio;
}

void updateMpcObservation(SystemObservation& mpcObservation,
                          const robot::model::RobotState& robotState,
                          const CentroidalMpcRobotModel<scalar_t>& mpcRobotModel,
                          PinocchioInterface& pinocchioInterface,
                          const std::vector<size_t>& mpcJointIndices,
                          ContinuousYawUnwrapper& yawUnwrapper,
                          double mpcTimeOffset = 0.0) {
  updateMpcState(mpcObservation.state, robotState, mpcRobotModel,
                 pinocchioInterface, mpcJointIndices, yawUnwrapper);
  mpcObservation.time = robotState.getTime() + mpcTimeOffset;
  mpcObservation.input = vector_t::Zero(mpcRobotModel.getInputDim());
  const std::vector<bool> configContacts = robotState.getContactFlags();
  contact_flag_t contactFlags = {true, true};
  if (configContacts.size() >= 2) {
    contactFlags[0] = configContacts[0];
    contactFlags[1] = configContacts[1];
  }
  mpcObservation.mode = stanceLeg2ModeNumber(contactFlags);
}

// Root Cause 1 & B Fix: High-authority joint tracking stiffness for single-support stance,
// swing-leg placement, and yaw turning without knee sag ("duck-walk") or leg crossing.
double trackingKp(const std::string& jointName) {
  if (jointName.find("hip_pitch")   != std::string::npos) return 400.0;
  if (jointName.find("hip_roll")    != std::string::npos) return 350.0;
  if (jointName.find("hip_yaw")     != std::string::npos) return 180.0;
  if (jointName.find("knee")        != std::string::npos) return 450.0;
  if (jointName.find("ankle_pitch") != std::string::npos) return 150.0;
  if (jointName.find("ankle_roll")  != std::string::npos) return  80.0;
  if (jointName.find("waist_pitch") != std::string::npos) return 200.0;
  if (jointName.find("waist_roll")  != std::string::npos) return 200.0;
  if (jointName.find("waist_yaw")   != std::string::npos) return 120.0;
  if (jointName.find("shoulder")    != std::string::npos) return  40.0;
  if (jointName.find("elbow")       != std::string::npos) return  30.0;
  return 25.0;
}

double trackingKd(const std::string& jointName) {
  if (jointName.find("hip_pitch")   != std::string::npos) return 25.0;
  if (jointName.find("hip_roll")    != std::string::npos) return 22.0;
  if (jointName.find("hip_yaw")     != std::string::npos) return 10.0;
  if (jointName.find("knee")        != std::string::npos) return 25.0;
  if (jointName.find("ankle_pitch") != std::string::npos) return 10.0;
  if (jointName.find("ankle_roll")  != std::string::npos) return  5.0;
  if (jointName.find("waist_pitch") != std::string::npos) return 15.0;
  if (jointName.find("waist_roll")  != std::string::npos) return 15.0;
  if (jointName.find("waist_yaw")   != std::string::npos) return 10.0;
  if (jointName.find("shoulder")    != std::string::npos) return  4.0;
  if (jointName.find("elbow")       != std::string::npos) return  3.0;
  return 2.0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> programArgs =
      rclcpp::remove_ros_arguments(argc, argv);
  if (programArgs.size() < 6) {
    throw std::runtime_error(
        "Usage: drake_mrt_sim_node robot task.info reference.info "
        "robot.urdf gait.info");
  }

  const std::string robotName = programArgs[1];
  const std::string taskFile = programArgs[2];
  const std::string referenceFile = programArgs[3];
  const std::string urdfFile = programArgs[4];
  const std::string gaitFile = programArgs[5];

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(robotName + "_drake_mrt_sim");

  node->declare_parameter<double>("sim_dt", 0.0005);
  node->declare_parameter<double>("control_rate_hz", 500.0);
  node->declare_parameter<double>("joint_damping", 10.0);
  node->declare_parameter<double>("ground_static_friction", 2.0);
  node->declare_parameter<double>("ground_dynamic_friction", 1.5);
  node->declare_parameter<double>("ground_dissipation", 2.0);
  node->declare_parameter<double>("ground_point_stiffness", 5.0e4);
  node->declare_parameter<bool>("force_mujoco_contact_flags", true);
  node->declare_parameter<bool>("reset_on_fall", true);
  node->declare_parameter<double>("fall_z_threshold", 0.2);
  node->declare_parameter<double>("reset_cooldown_s", 1.0);
  node->declare_parameter<double>("max_forward_vel_x", 0.65);
  node->declare_parameter<double>("max_backward_vel_x", 0.48);
  node->declare_parameter<double>("max_lateral_vel_y", 0.30);
  node->declare_parameter<double>("max_yaw_rate_z", 0.32);
  node->declare_parameter<double>("min_pelvis_height", 0.68);
  node->declare_parameter<double>("max_pelvis_height", 0.84);
  node->declare_parameter<bool>("enable_meshcat", true);
  node->declare_parameter<std::string>("meshcat_host", "*");
  node->declare_parameter<double>("meshcat_publish_rate_hz", 30.0);
  node->declare_parameter<bool>("verbose", true);

  // Model & Pinocchio interfaces (setupOCP = false for MRT follower sim node)
  CentroidalMpcInterface interface(taskFile, urdfFile, referenceFile, false);
  auto pinocchioInterface = interface.getPinocchioInterface();
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> mpcModelPtr(interface.getMpcRobotModel().clone());
  auto& mpcModel = *mpcModelPtr;
  const auto& modelSettings = interface.modelSettings();

  robot::drake_sim_interface::DrakeSimConfig simConfig;
  auto initialState = std::make_shared<robot::model::RobotState>(
      makeInitialState(urdfFile, interface));
  simConfig.initStatePtr = initialState;
  simConfig.dt = node->get_parameter("sim_dt").as_double();
  simConfig.jointDamping = node->get_parameter("joint_damping").as_double();
  simConfig.groundStaticFriction = node->get_parameter("ground_static_friction").as_double();
  simConfig.groundDynamicFriction = node->get_parameter("ground_dynamic_friction").as_double();
  simConfig.groundDissipation = node->get_parameter("ground_dissipation").as_double();
  simConfig.groundPointStiffness = node->get_parameter("ground_point_stiffness").as_double();
  simConfig.forceContactFlags = node->get_parameter("force_mujoco_contact_flags").as_bool();
  simConfig.resetOnFall = node->get_parameter("reset_on_fall").as_bool();
  simConfig.fallHeight = node->get_parameter("fall_z_threshold").as_double();
  simConfig.resetPause = node->get_parameter("reset_cooldown_s").as_double();
  simConfig.enableVisualization = node->get_parameter("enable_meshcat").as_bool();
  simConfig.meshcatHost = node->get_parameter("meshcat_host").as_string();
  simConfig.verbose = node->get_parameter("verbose").as_bool();

  robot::drake_sim_interface::DrakeSimInterface robotInterface(simConfig, urdfFile);

  const auto mpcJointIndices =
      robotInterface.getRobotDescription().getJointIndices(modelSettings.mpcModelJointNames);
  const auto fixedJointIndices =
      robotInterface.getRobotDescription().getJointIndices(modelSettings.fixedJointNames);

  std::vector<double> mpcJointKp(mpcJointIndices.size());
  std::vector<double> mpcJointKd(mpcJointIndices.size());
  for (size_t i = 0; i < mpcJointIndices.size(); ++i) {
    mpcJointKp[i] = trackingKp(modelSettings.mpcModelJointNames[i]);
    mpcJointKd[i] = trackingKd(modelSettings.mpcModelJointNames[i]);
  }

  std::vector<double> fixedJointKp(fixedJointIndices.size());
  std::vector<double> fixedJointKd(fixedJointIndices.size());
  for (size_t i = 0; i < fixedJointIndices.size(); ++i) {
    fixedJointKp[i] = trackingKp(modelSettings.fixedJointNames[i]);
    fixedJointKd[i] = trackingKd(modelSettings.fixedJointNames[i]);
  }

  // Target trajectories calculator for reset messages
  CentroidalMpcTargetTrajectoriesCalculator targetCalculator(
      referenceFile, mpcModel, pinocchioInterface,
      interface.getCentroidalModelInfo(), interface.mpcSettings().timeHorizon_);
  const vector_t& initMpcState = interface.getInitialState();
  const double nominalPelvisZ = mpcModel.getBasePosition(initMpcState).z();

  // Root Cause 3 Fix: Read reference.info velocity multipliers and configure safe
  // velocity command scaling/clamping bridge (/humanoid/walking_velocity_command ->
  // /humanoid/walking_velocity_command_clamped).
  scalar_t refMaxVelX = 2.4;
  scalar_t refMaxVelY = 1.2;
  scalar_t refMaxVelYaw = 1.0;
  loadData::loadCppDataType(referenceFile, "maxDisplacementVelocityX", refMaxVelX);
  loadData::loadCppDataType(referenceFile, "maxDisplacementVelocityY", refMaxVelY);
  loadData::loadCppDataType(referenceFile, "maxRotationVelocity", refMaxVelYaw);

  const double maxForwardVelX = std::max(0.1, node->get_parameter("max_forward_vel_x").as_double());
  const double maxBackwardVelX = std::max(0.1, node->get_parameter("max_backward_vel_x").as_double());
  const double maxLateralVelY = std::max(0.05, node->get_parameter("max_lateral_vel_y").as_double());
  const double maxYawRateZ = std::max(0.05, node->get_parameter("max_yaw_rate_z").as_double());
  const double minPelvisHeight = node->get_parameter("min_pelvis_height").as_double();
  const double maxPelvisHeight = node->get_parameter("max_pelvis_height").as_double();

  auto humanoidVisualizer = std::make_shared<HumanoidVisualizer>(
      taskFile, interface.getPinocchioInterface(),
      interface.getMpcRobotModel(), node);

  // Instantiate MRT ROS Interface
  auto qos = rclcpp::QoS(1).best_effort();
  MRT_ROS_Interface mrt(robotName);
  mrt.launchNodes(node, qos);

  auto clampedVelPub =
      node->create_publisher<humanoid_mpc_msgs::msg::WalkingVelocityCommand>(
          "/humanoid/walking_velocity_command_clamped", qos);

  std::mutex velCmdMutex;
  bool blockExternalVelCmds = false;
  double postResetHoldUntilSimTime = -1.0;
  bool wasActivelyWalking = false;
  bool stopAlignmentPendingZero = false;
  auto stopAlignUntilWallTime = std::chrono::steady_clock::time_point::min();

  auto publishZeroVelocityCommand = [&]() {
    humanoid_mpc_msgs::msg::WalkingVelocityCommand zeroMsg;
    zeroMsg.linear_velocity_x = 0.0;
    zeroMsg.linear_velocity_y = 0.0;
    zeroMsg.angular_velocity_z = 0.0;
    zeroMsg.desired_pelvis_height = nominalPelvisZ;
    clampedVelPub->publish(zeroMsg);
  };

  auto rawVelSub =
      node->create_subscription<humanoid_mpc_msgs::msg::WalkingVelocityCommand>(
          "/humanoid/walking_velocity_command", rclcpp::QoS(10).best_effort(),
          [&](const humanoid_mpc_msgs::msg::WalkingVelocityCommand::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(velCmdMutex);
            if (blockExternalVelCmds) {
              wasActivelyWalking = false;
              stopAlignmentPendingZero = false;
              publishZeroVelocityCommand();
              return;
            }
            const double rawVx = std::isfinite(msg->linear_velocity_x)
                                     ? std::clamp(msg->linear_velocity_x, -1.0, 1.0)
                                     : 0.0;
            const double rawVy = std::isfinite(msg->linear_velocity_y)
                                     ? std::clamp(msg->linear_velocity_y, -1.0, 1.0)
                                     : 0.0;
            const double rawWz = std::isfinite(msg->angular_velocity_z)
                                     ? std::clamp(msg->angular_velocity_z, -1.0, 1.0)
                                     : 0.0;
            const double rawH = (std::isfinite(msg->desired_pelvis_height) &&
                                 msg->desired_pelvis_height > 0.1)
                                    ? msg->desired_pelvis_height
                                    : nominalPelvisZ;

            const bool isZeroCmd = (std::abs(rawVx) < 0.02 &&
                                    std::abs(rawVy) < 0.02 &&
                                    std::abs(rawWz) < 0.02);
            const auto nowWall = std::chrono::steady_clock::now();

            if (!isZeroCmd) {
              if (std::abs(rawVx) >= 0.05 || std::abs(rawVy) >= 0.05 || std::abs(rawWz) >= 0.05) {
                wasActivelyWalking = true;
              }
              stopAlignmentPendingZero = false;
            } else if (wasActivelyWalking) {
              wasActivelyWalking = false;
              stopAlignmentPendingZero = true;
              stopAlignUntilWallTime = nowWall + std::chrono::milliseconds(750);
            }

            // Smooth soft-saturation mapping raw [-1, 1] command to safe physical velocities
            double physVx =
                (rawVx >= 0.0)
                    ? (maxForwardVelX * std::tanh(rawVx / 0.42))
                    : (maxBackwardVelX * std::tanh(rawVx / 0.35));
            double physVy = maxLateralVelY * std::tanh(rawVy / 0.50);
            const double linearLoad = std::clamp(std::hypot(rawVx, rawVy), 0.0, 1.0);
            const double yawTurnScale = 1.0 - 0.25 * linearLoad;
            double physWz = maxYawRateZ * yawTurnScale * std::tanh(rawWz / 0.55);
            double safeH = std::clamp(rawH, minPelvisHeight, maxPelvisHeight);

            // Root Cause B Fix: Hold a brief 0.75 s slow_walk centering window (physVx = 0.055 m/s)
            // when returning to Center so the feet step underneath the hips at nominalPelvisZ
            // before ModeNumber::STANCE locks both feet in place.
            if (isZeroCmd && stopAlignmentPendingZero) {
              if (nowWall < stopAlignUntilWallTime) {
                physVx = 0.055;
                physVy = 0.0;
                physWz = 0.0;
                safeH = nominalPelvisZ;
              } else {
                stopAlignmentPendingZero = false;
                physVx = 0.0;
                physVy = 0.0;
                physWz = 0.0;
                safeH = nominalPelvisZ;
              }
            }

            humanoid_mpc_msgs::msg::WalkingVelocityCommand outMsg;
            outMsg.linear_velocity_x =
                (std::abs(refMaxVelX) > 1e-6) ? (physVx / refMaxVelX) : physVx;
            outMsg.linear_velocity_y =
                (std::abs(refMaxVelY) > 1e-6) ? (physVy / refMaxVelY) : physVy;
            outMsg.angular_velocity_z =
                (std::abs(refMaxVelYaw) > 1e-6) ? (physWz / refMaxVelYaw) : physWz;
            outMsg.desired_pelvis_height = safeH;
            clampedVelPub->publish(outMsg);
          });
  (void)rawVelSub;

  // =========================================================================
  // STEP 1: Hold G1 in Nominal Upright Pose in Drake Without Advancing Physics
  // =========================================================================
  RCLCPP_INFO(node->get_logger(),
              "[STARTUP STEP 1/5] Holding G1 in nominal pose in Drake (physics held at t=0)...");
  robotInterface.initSim();
  robotInterface.updateInterfaceStateFromRobot();

  // Pre-populate nominal joint holding PD
  auto& jointAction = robotInterface.getRobotJointAction();
  for (size_t i = 0; i < mpcJointIndices.size(); ++i) {
    const size_t jIdx = mpcJointIndices[i];
    auto& act = jointAction.at(jIdx).value();
    act.q_des = initialState->getJointPosition(jIdx);
    act.qd_des = 0.0;
    act.kp = mpcJointKp[i];
    act.kd = mpcJointKd[i];
    act.feed_forward_effort = 0.0;
  }
  for (size_t i = 0; i < fixedJointIndices.size(); ++i) {
    const size_t jIdx = fixedJointIndices[i];
    auto& act = jointAction.at(jIdx).value();
    act.q_des = initialState->getJointPosition(jIdx);
    act.qd_des = 0.0;
    act.kp = fixedJointKp[i];
    act.kd = fixedJointKd[i];
    act.feed_forward_effort = 0.0;
  }
  robotInterface.applyJointAction();

  // =========================================================================
  // STEP 2: Call /g1/mpc_reset Service on Node 1
  // =========================================================================
  RCLCPP_INFO(node->get_logger(),
              "[STARTUP STEP 2/5] Calling /%s/mpc_reset service with initial stance trajectory...",
              robotName.c_str());
  const vector4_t initialVelTarget(0.0, 0.0, nominalPelvisZ, 0.0);
  auto initialTargetTrajectories =
      targetCalculator.commandedVelocityToTargetTrajectories(initialVelTarget, 0.0, initMpcState);
  mrt.resetMpcNode(initialTargetTrajectories);

  // =========================================================================
  // STEP 3: Run Observation Warm-Up Loop (30 Cycles at 100 Hz)
  // =========================================================================
  RCLCPP_INFO(node->get_logger(),
              "[STARTUP STEP 3/5] Priming Node 1 static alpha-filter (30 cycles @ 100 Hz)...");
  SystemObservation initObs;
  initObs.time = 0.0;
  initObs.state = initMpcState;
  initObs.input = vector_t::Zero(mpcModel.getInputDim());
  initObs.mode = ModeNumber::STANCE;

  for (int cycle = 0; cycle < 30 && rclcpp::ok(); ++cycle) {
    mrt.setCurrentObservation(initObs);
    mrt.spinMRT();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    mrt.updatePolicy();
  }

  // =========================================================================
  // STEP 4: Verify Receipt of Valid MPC Policy (Knot Height >= 0.78 m)
  // =========================================================================
  RCLCPP_INFO(node->get_logger(),
              "[STARTUP STEP 4/5] Verifying received policy knot height >= 0.78 m...");
  bool policyReady = false;
  const auto verifyTimeout = std::chrono::steady_clock::now() + std::chrono::seconds(15);

  while (rclcpp::ok() && std::chrono::steady_clock::now() < verifyTimeout) {
    mrt.setCurrentObservation(initObs);
    mrt.spinMRT();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    mrt.updatePolicy();

    if (mrt.initialPolicyReceived()) {
      const auto& policy = mrt.getPolicy();
      if (!policy.stateTrajectory_.empty()) {
        const double knotFrontZ = mpcModel.getBasePosition(policy.stateTrajectory_.front()).z();
        const double knotBackZ = mpcModel.getBasePosition(policy.stateTrajectory_.back()).z();
        if (knotFrontZ >= 0.78 && knotBackZ >= 0.78) {
          policyReady = true;
          RCLCPP_INFO(node->get_logger(),
                      "[STARTUP STEP 4/5 PASSED] Valid policy verified! Front=%.4f m, Back=%.4f m (>= 0.78 m)",
                      knotFrontZ, knotBackZ);
          break;
        } else {
          RCLCPP_INFO_THROTTLE(node->get_logger(), *node->get_clock(), 500,
                               "[STARTUP WARMUP] Waiting for filter convergence: Front=%.4f m, Back=%.4f m (target >= 0.78 m)",
                               knotFrontZ, knotBackZ);
        }
      }
    }
  }
  if (!policyReady) {
    throw std::runtime_error("Startup synchronization failed: Node 1 policy knot height < 0.78 m");
  }

  // =========================================================================
  // STEP 5: Release 2000 Hz Physics Stepping & Enter Active Control Loop
  // =========================================================================
  RCLCPP_INFO(node->get_logger(),
              "[STARTUP STEP 5/5] Releasing 2000 Hz physics simulation & entering 500 Hz follower loop...");

  // Pre-evaluate policy at t = 0.005 s for initial feedforward holding torques
  vector_t mpcPolicyState, mpcPolicyInput;
  size_t mpcPolicyMode;
  mrt.evaluatePolicy(0.005, initMpcState, mpcPolicyState, mpcPolicyInput, mpcPolicyMode);

  vector_t mpc_q_des = mpcModel.getJointAngles(mpcPolicyState);
  vector_t mpc_qd_des = mpcModel.getJointVelocities(mpcPolicyState, mpcPolicyInput);
  vector_t qdd_des = vector_t::Zero(mpcModel.getJointDim());
  std::array<vector6_t, 2> footWrenches = {
      mpcModel.getContactWrench(mpcPolicyInput, 0),
      mpcModel.getContactWrench(mpcPolicyInput, 1)};

  vector_t q = mpcModel.getGeneralizedCoordinates(initMpcState);
  vector_t qd = mpcModel.getGeneralizedVelocities(initMpcState, initObs.input);
  vector_t ffTorques = computeJointTorques<scalar_t>(q, qd, qdd_des, footWrenches, pinocchioInterface);

  for (size_t i = 0; i < mpcJointIndices.size(); ++i) {
    const size_t jIdx = mpcJointIndices[i];
    auto& act = jointAction.at(jIdx).value();
    act.q_des = mpc_q_des[i];
    act.qd_des = mpc_qd_des[i];
    act.kp = mpcJointKp[i];
    act.kd = mpcJointKd[i];
    act.feed_forward_effort =
        ffTorques[i] + simConfig.jointDamping * mpc_qd_des[i];
  }
  for (size_t i = 0; i < fixedJointIndices.size(); ++i) {
    const size_t jIdx = fixedJointIndices[i];
    auto& act = jointAction.at(jIdx).value();
    act.q_des = initialState->getJointPosition(jIdx);
    act.qd_des = 0.0;
    act.kp = fixedJointKp[i];
    act.kd = fixedJointKd[i];
    act.feed_forward_effort = 0.0;
  }
  robotInterface.applyJointAction();

  // START 2000 Hz Drake Physics Thread
  robotInterface.startSim();

  RCLCPP_INFO(node->get_logger(),
              "Drake URDF simulation started: dt=%.6f s, control=%.1f Hz; "
              "MRT is the only joint controller",
              simConfig.dt, node->get_parameter("control_rate_hz").as_double());
  if (simConfig.enableVisualization) {
    RCLCPP_INFO(node->get_logger(), "Meshcat: %s",
                robotInterface.getVisualizationUrl().c_str());
    RCLCPP_INFO(node->get_logger(), "Meshcat server scene: %s",
                robotInterface.getVisualizationStatus().c_str());
  }

  // Control loop timing
  const double controlRateHz = node->get_parameter("control_rate_hz").as_double();
  const auto controlPeriod = std::chrono::microseconds(
      std::max<int64_t>(1, static_cast<int64_t>(std::llround(1.0e6 / controlRateHz))));
  const double meshcatRateHz = node->get_parameter("meshcat_publish_rate_hz").as_double();
  const auto meshcatPeriod = std::chrono::microseconds(
      meshcatRateHz > 0.0
          ? std::max<int64_t>(1, static_cast<int64_t>(std::llround(1.0e6 / meshcatRateHz)))
          : 1);

  uint64_t lastFallCount = robotInterface.getFallResetCount();
  auto nextIteration = std::chrono::steady_clock::now();
  auto nextVisualization = nextIteration;
  SystemObservation currentObs;
  currentObs.state = initMpcState;
  currentObs.input = vector_t::Zero(mpcModel.getInputDim());

  const size_t obsDecimation = std::max<size_t>(
      1, static_cast<size_t>(std::round(controlRateHz / 40.0)));
  size_t loopIteration = 0;
  double lastTelemetrySimTime = -1.0;
  double mpcTimeOffset = 0.0;
  ContinuousYawUnwrapper yawUnwrapper;

  while (rclcpp::ok()) {
    nextIteration += controlPeriod;
    const uint64_t currentFallCount = robotInterface.getFallResetCount();

    // =======================================================================
    // Fall Detection & Recovery Reset (Root Cause 2 Fix)
    // =======================================================================
    if (currentFallCount > lastFallCount) {
      const auto recoveryWallStart = std::chrono::steady_clock::now();
      RCLCPP_WARN(node->get_logger(),
                  "[FALL RECOVERY] Fall detected (%lu -> %lu). Executing synchronized recovery protocol...",
                  lastFallCount, currentFallCount);
      lastFallCount = currentFallCount;
      loopIteration = 0;
      lastTelemetrySimTime = -1.0;
      yawUnwrapper.reset();

      robotInterface.updateInterfaceStateFromRobot();
      const double resetSimTime = robotInterface.getRobotState().getTime();

      {
        std::lock_guard<std::mutex> lock(velCmdMutex);
        blockExternalVelCmds = true;
        wasActivelyWalking = false;
        stopAlignmentPendingZero = false;
        postResetHoldUntilSimTime = resetSimTime + 1.0;
        publishZeroVelocityCommand();
      }

      // Phase A: Flush Node 1 velocityCommandFilter & step ProceduralMpcMotionManager
      // down to ModeNumber::STANCE by advancing mpcTimeOffset across 25 cycles.
      SystemObservation resetObs;
      resetObs.state = initMpcState;
      resetObs.input = vector_t::Zero(mpcModel.getInputDim());
      resetObs.mode = ModeNumber::STANCE;
      for (int c = 0; c < 25 && rclcpp::ok(); ++c) {
        publishZeroVelocityCommand();
        mpcTimeOffset += 0.08;
        resetObs.time = resetSimTime + mpcTimeOffset;
        mrt.setCurrentObservation(resetObs);
        mrt.spinMRT();
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        mrt.updatePolicy();
      }

      // Phase B: Reset Node 1 reference trajectory & solver cache at synchronized time
      resetObs.time = resetSimTime + mpcTimeOffset;
      auto resetTargetTrajectories =
          targetCalculator.commandedVelocityToTargetTrajectories(
              initialVelTarget, resetObs.time, initMpcState);
      mrt.resetMpcNode(resetTargetTrajectories);

      // Phase C: Re-warmup observation burst & verify STANCE mode + knot height >= 0.78 m
      for (int c = 0; c < 20 && rclcpp::ok(); ++c) {
        publishZeroVelocityCommand();
        mrt.setCurrentObservation(resetObs);
        mrt.spinMRT();
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        mrt.updatePolicy();
        if (mrt.initialPolicyReceived()) {
          const auto& policy = mrt.getPolicy();
          if (!policy.stateTrajectory_.empty()) {
            const double frontZ = mpcModel.getBasePosition(policy.stateTrajectory_.front()).z();
            const double backZ = mpcModel.getBasePosition(policy.stateTrajectory_.back()).z();
            const bool allStance = std::all_of(
                policy.modeSchedule_.modeSequence.begin(),
                policy.modeSchedule_.modeSequence.end(),
                [](size_t m) { return m == ModeNumber::STANCE; });
            if (c >= 10 && frontZ >= 0.78 && backZ >= 0.78 && allStance) {
              break;
            }
          }
        }
      }

      // Synchronize with remaining DrakeSimInterface reset_cooldown_s before releasing control
      const auto cooldownDeadline =
          recoveryWallStart +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<double>(std::max(0.0, simConfig.resetPause - 0.05)));
      while (rclcpp::ok() && std::chrono::steady_clock::now() < cooldownDeadline) {
        publishZeroVelocityCommand();
        mrt.setCurrentObservation(resetObs);
        mrt.spinMRT();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        mrt.updatePolicy();
      }

      // Re-apply fresh holding action right as physics resumes
      robotInterface.updateInterfaceStateFromRobot();
      updateMpcObservation(currentObs, robotInterface.getRobotState(), mpcModel,
                           pinocchioInterface, mpcJointIndices, yawUnwrapper,
                           mpcTimeOffset);
      mrt.evaluatePolicy(currentObs.time + 0.005, currentObs.state, mpcPolicyState,
                         mpcPolicyInput, mpcPolicyMode);
      mpc_q_des = mpcModel.getJointAngles(mpcPolicyState);
      mpc_qd_des = mpcModel.getJointVelocities(mpcPolicyState, mpcPolicyInput);
      qdd_des.setZero();
      footWrenches = {mpcModel.getContactWrench(mpcPolicyInput, 0),
                      mpcModel.getContactWrench(mpcPolicyInput, 1)};
      q = mpcModel.getGeneralizedCoordinates(currentObs.state);
      qd = mpcModel.getGeneralizedVelocities(currentObs.state, currentObs.input);
      ffTorques = computeJointTorques<scalar_t>(q, qd, qdd_des, footWrenches, pinocchioInterface);

      for (size_t i = 0; i < mpcJointIndices.size(); ++i) {
        const size_t jIdx = mpcJointIndices[i];
        auto& act = jointAction.at(jIdx).value();
        act.q_des = mpc_q_des[i];
        act.qd_des = mpc_qd_des[i];
        act.kp = mpcJointKp[i];
        act.kd = mpcJointKd[i];
        act.feed_forward_effort =
            ffTorques[i] + simConfig.jointDamping * mpc_qd_des[i];
      }
      for (size_t i = 0; i < fixedJointIndices.size(); ++i) {
        const size_t jIdx = fixedJointIndices[i];
        auto& act = jointAction.at(jIdx).value();
        act.q_des = initialState->getJointPosition(jIdx);
        act.qd_des = 0.0;
        act.kp = fixedJointKp[i];
        act.kd = fixedJointKd[i];
        act.feed_forward_effort = 0.0;
      }
      robotInterface.applyJointAction();
      nextIteration = std::chrono::steady_clock::now();
      RCLCPP_INFO(node->get_logger(), "[FALL RECOVERY] Stance re-established.");
    } else {
      // =====================================================================
      // Active 500 Hz Follower Loop
      // =====================================================================
      robotInterface.updateInterfaceStateFromRobot();
      const auto& currentRobotState = robotInterface.getRobotState();
      updateMpcObservation(currentObs, currentRobotState, mpcModel,
                           pinocchioInterface, mpcJointIndices, yawUnwrapper,
                           mpcTimeOffset);

      // Release post-reset velocity command hold once robot has stood stably for 1.0 s
      {
        std::lock_guard<std::mutex> lock(velCmdMutex);
        if (blockExternalVelCmds) {
          if (currentRobotState.getTime() >= postResetHoldUntilSimTime) {
            blockExternalVelCmds = false;
          } else if (loopIteration % obsDecimation == 0) {
            publishZeroVelocityCommand();
          }
        } else if (stopAlignmentPendingZero &&
                   std::chrono::steady_clock::now() >= stopAlignUntilWallTime) {
          stopAlignmentPendingZero = false;
          publishZeroVelocityCommand();
        }
      }

      // Publish observation to Node 1 decimated to ~40 Hz (avoids SQP queue backlog)
      if (loopIteration % obsDecimation == 0) {
        mrt.setCurrentObservation(currentObs);
      }
      ++loopIteration;

      // Spin MRT callbacks & update policy buffer
      mrt.spinMRT();
      mrt.updatePolicy();

      // Evaluate active policy at t + 0.005 s and t + 0.015 s for finite-difference qdd_des
      constexpr double kEvalLookaheadSec = 0.005;
      constexpr double kAccelFiniteDiffDt = 0.010;
      constexpr double kMaxJointAccel = 35.0;
      mrt.evaluatePolicy(currentObs.time + kEvalLookaheadSec, currentObs.state,
                         mpcPolicyState, mpcPolicyInput, mpcPolicyMode);
      vector_t mpcPolicyStateAhead, mpcPolicyInputAhead;
      size_t mpcPolicyModeAhead = mpcPolicyMode;
      mrt.evaluatePolicy(currentObs.time + kEvalLookaheadSec + kAccelFiniteDiffDt,
                         currentObs.state, mpcPolicyStateAhead, mpcPolicyInputAhead,
                         mpcPolicyModeAhead);

      mpc_q_des = mpcModel.getJointAngles(mpcPolicyState);
      mpc_qd_des = mpcModel.getJointVelocities(mpcPolicyState, mpcPolicyInput);
      const vector_t mpc_qd_des_ahead =
          mpcModel.getJointVelocities(mpcPolicyStateAhead, mpcPolicyInputAhead);
      for (Eigen::Index j = 0; j < qdd_des.size(); ++j) {
        const double raw_qdd = (mpc_qd_des_ahead[j] - mpc_qd_des[j]) / kAccelFiniteDiffDt;
        qdd_des[j] = std::isfinite(raw_qdd)
                         ? std::clamp(raw_qdd, -kMaxJointAccel, kMaxJointAccel)
                         : 0.0;
      }
      footWrenches = {mpcModel.getContactWrench(mpcPolicyInput, 0),
                      mpcModel.getContactWrench(mpcPolicyInput, 1)};
      q = mpcModel.getGeneralizedCoordinates(currentObs.state);
      qd = mpcModel.getGeneralizedVelocities(currentObs.state, currentObs.input);
      ffTorques = computeJointTorques<scalar_t>(q, qd, qdd_des, footWrenches, pinocchioInterface);

      for (size_t i = 0; i < mpcJointIndices.size(); ++i) {
        const size_t jIdx = mpcJointIndices[i];
        auto& act = jointAction.at(jIdx).value();
        act.q_des = mpc_q_des[i];
        act.qd_des = mpc_qd_des[i];
        act.kp = mpcJointKp[i];
        act.kd = mpcJointKd[i];
        act.feed_forward_effort =
            ffTorques[i] + simConfig.jointDamping * mpc_qd_des[i];
      }
      for (size_t i = 0; i < fixedJointIndices.size(); ++i) {
        const size_t jIdx = fixedJointIndices[i];
        auto& act = jointAction.at(jIdx).value();
        act.q_des = initialState->getJointPosition(jIdx);
        act.qd_des = 0.0;
        act.kp = fixedJointKp[i];
        act.kd = fixedJointKd[i];
        act.feed_forward_effort = 0.0;
      }
      robotInterface.applyJointAction();
    }

    // Standing Telemetry Logging at 2 Hz
    const auto& telemetryState = robotInterface.getRobotState();
    const auto pelvisPos = telemetryState.getRootPositionInWorldFrame();
    const auto pelvisRot = telemetryState.getRootRotationLocalToWorldFrame();
    const double qx = pelvisRot.x(), qy = pelvisRot.y(), qz = pelvisRot.z(), qw = pelvisRot.w();
    const double roll = std::atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy));
    const double pitch = std::asin(std::clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0));
    const double simTime = telemetryState.getTime();
    if (lastTelemetrySimTime < 0.0 || (simTime - lastTelemetrySimTime) >= 0.5 || simTime < lastTelemetrySimTime) {
      lastTelemetrySimTime = simTime;
      RCLCPP_INFO(
          node->get_logger(),
          "[standing_telemetry] t_sim=%.3f s | pelvis_z=%.4f m | roll=%.4f rad | pitch=%.4f rad | falls=%lu",
          simTime, pelvisPos.z(), roll, pitch,
          robotInterface.getFallResetCount());
      RCLCPP_INFO(
          node->get_logger(),
          "[walking_telemetry] t_sim=%.3f s | pelvis_x=%.4f m | pelvis_z=%.4f m | falls=%lu",
          simTime, pelvisPos.x(), pelvisPos.z(),
          robotInterface.getFallResetCount());
    }

    // Meshcat & RViz Visualization
    const auto now = std::chrono::steady_clock::now();
    if (meshcatRateHz > 0.0 && now >= nextVisualization) {
      try {
        if (simConfig.enableVisualization) {
          robotInterface.publishVisualization();
        }
        if (humanoidVisualizer && mrt.initialPolicyReceived()) {
          humanoidVisualizer->update(currentObs, mrt.getPolicy(), mrt.getCommand());
        }
      } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(
            node->get_logger(), *node->get_clock(), 2000,
            "[VISUALIZATION] Exception during visualization update: %s", e.what());
      }
      nextVisualization = now + meshcatPeriod;
    }

    std::this_thread::sleep_until(nextIteration);
    if (std::chrono::steady_clock::now() > nextIteration + controlPeriod) {
      nextIteration = std::chrono::steady_clock::now();
    }
  }

  mrt.shutdownNodes();
  rclcpp::shutdown();
  return 0;
}
