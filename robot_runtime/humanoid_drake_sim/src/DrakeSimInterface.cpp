#include "humanoid_drake_sim/DrakeSimInterface.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <drake/geometry/collision_filter_declaration.h>
#include <drake/geometry/proximity_properties.h>
#include <drake/geometry/meshcat.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/shape_specification.h>
#include <drake/math/rigid_transform.h>
#include <drake/multibody/math/spatial_algebra.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/contact_results.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/revolute_joint.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram.h>
#include <drake/systems/framework/diagram_builder.h>

namespace robot::drake_sim_interface {
namespace mbd = drake::multibody;
namespace geometry = drake::geometry;
namespace systems = drake::systems;

namespace {

void configurePackageMap(mbd::Parser* parser) {
  parser->package_map().PopulateFromEnvironment("AMENT_PREFIX_PATH");
  try {
    const std::string share =
        ament_index_cpp::get_package_share_directory("g1_description");
    if (!parser->package_map().Contains("g1_description")) {
      parser->package_map().Add("g1_description", share);
    }
  } catch (const std::exception&) {
    // Non-G1 URDFs can resolve meshes through AMENT_PREFIX_PATH or absolute
    // paths, so this optional package lookup is not fatal.
  }
}

}  // namespace

struct DrakeSimInterface::Impl {
  mbd::MultibodyPlant<double>* plant{nullptr};
  mbd::ModelInstanceIndex modelInstance;
  std::unique_ptr<systems::Diagram<double>> diagram;
  std::unique_ptr<systems::Simulator<double>> simulator;
  std::vector<std::string> actuatorJointNames;
  std::vector<joint_index_t> robotActuatorIndices;
  std::vector<double> effortLimits;
  Eigen::VectorXd actuation;

  mbd::MultibodyPlant<double>* visualizationPlant{nullptr};
  mbd::ModelInstanceIndex visualizationModelInstance;
  std::shared_ptr<geometry::Meshcat> meshcat;
  std::unique_ptr<systems::Diagram<double>> visualizationDiagram;
  std::unique_ptr<systems::Context<double>> visualizationContext;
  std::string visualizationRobotRootPath;
  std::atomic<uint64_t> fallResetCount{0};
};

DrakeSimInterface::DrakeSimInterface(const DrakeSimConfig& config,
                                     const std::string& urdfPath)
    : RobotHWInterfaceBase(urdfPath),
      config_(config),
      robotStateInternal_(getRobotDescription(), 2),
      robotJointActionInternal_(getRobotDescription()),
      initialState_(getRobotDescription(), 2),
      impl_(std::make_unique<Impl>()) {
  if (!(config_.dt > 0.0)) {
    throw std::invalid_argument("Drake simulation dt must be positive");
  }

  if (config_.initStatePtr) {
    initialState_ = *config_.initStatePtr;
  } else {
    initialState_.setConfigurationToZero();
    initialState_.setRootPositionInWorldFrame(vector3_t(0.0, 0.0, 1.0));
  }

  buildPlant(urdfPath);
  if (config_.enableVisualization) {
    buildVisualization(urdfPath);
  }
  setupJointIndexMaps();
  setSimState(initialState_);
  impl_->plant->get_actuation_input_port(impl_->modelInstance)
      .FixValue(&impl_->diagram->GetMutableSubsystemContext(
                    *impl_->plant, &impl_->simulator->get_mutable_context()),
                impl_->actuation);
  impl_->simulator->Initialize();
  updateThreadSafeRobotState();
  updateInterfaceStateFromRobot();
  publishVisualization();
}

DrakeSimInterface::~DrakeSimInterface() {
  terminate_.store(true);
  if (simulationThread_.joinable()) {
    simulationThread_.join();
  }
}

void DrakeSimInterface::buildPlant(const std::string& urdfPath) {
  systems::DiagramBuilder<double> builder;
  auto plantAndSceneGraph = mbd::AddMultibodyPlantSceneGraph(&builder, config_.dt);
  impl_->plant = &plantAndSceneGraph.plant;

  // This selects Drake's contact time-stepping algorithm only. It is not the
  // removed SAP joint-PD controller and does not produce commanded torques.
  impl_->plant->set_contact_model(mbd::ContactModel::kPointContactOnly);
  impl_->plant->set_discrete_contact_approximation(
      mbd::DiscreteContactApproximation::kSap);

  mbd::Parser parser(impl_->plant);
  configurePackageMap(&parser);
  const auto models = parser.AddModels(urdfPath);
  if (models.size() != 1) {
    throw std::runtime_error("Expected exactly one robot model in URDF: " +
                             urdfPath);
  }
  impl_->modelInstance = models.front();

  // This URDF intentionally has no ros_control transmission tags. Drake does
  // not create actuators from <limit effort=...>, so expose every joint in
  // RobotDescription as a torque actuator by name.
  for (const std::string& jointName : getRobotDescription().getJointNames()) {
    auto& joint = impl_->plant->GetMutableJointByName<mbd::RevoluteJoint>(
        jointName, impl_->modelInstance);
    joint.set_default_damping(config_.jointDamping);
    if (!impl_->plant->HasJointActuatorNamed(jointName,
                                             impl_->modelInstance)) {
      const double effortLimit =
          getRobotDescription().getJointDescription(jointName).max_effort;
      impl_->plant->AddJointActuator(jointName, joint, effortLimit);
    }
  }

  geometry::ProximityProperties groundProperties;
  geometry::AddContactMaterial(
      config_.groundDissipation, config_.groundPointStiffness,
      mbd::CoulombFriction<double>(config_.groundStaticFriction,
                                   config_.groundDynamicFriction),
      &groundProperties);
  impl_->plant->RegisterCollisionGeometry(
      impl_->plant->world_body(),
      drake::math::RigidTransformd(Eigen::Vector3d(0.0, 0.0, -0.1)),
      geometry::Box(20.0, 20.0, 0.2), "ground_collision",
      groundProperties);
  impl_->plant->RegisterVisualGeometry(
      impl_->plant->world_body(),
      drake::math::RigidTransformd(Eigen::Vector3d(0.0, 0.0, -0.1)),
      geometry::Box(20.0, 20.0, 0.2), "ground_visual",
      Eigen::Vector4d(0.5, 0.5, 0.5, 1.0));

  // Exclude all self-collisions within the robot model instance, matching MuJoCo
  // contype="0" conaffinity="0" so only foot spheres interact with the ground plane.
  std::vector<const mbd::RigidBody<double>*> bodies;
  bodies.reserve(impl_->plant->num_bodies());
  for (const auto& bodyIndex : impl_->plant->GetBodyIndices(impl_->modelInstance)) {
    bodies.push_back(&impl_->plant->get_body(bodyIndex));
  }
  drake::geometry::CollisionFilterDeclaration filter;
  filter.ExcludeWithin(impl_->plant->CollectRegisteredGeometries(bodies));
  plantAndSceneGraph.scene_graph.collision_filter_manager().Apply(filter);

  impl_->plant->Finalize();
  impl_->diagram = builder.Build();
  impl_->simulator =
      std::make_unique<systems::Simulator<double>>(*impl_->diagram);
  impl_->simulator->set_target_realtime_rate(0.0);
}

void DrakeSimInterface::buildVisualization(const std::string& urdfPath) {
  systems::DiagramBuilder<double> builder;
  auto plantAndSceneGraph = mbd::AddMultibodyPlantSceneGraph(&builder, 0.0);
  impl_->visualizationPlant = &plantAndSceneGraph.plant;

  mbd::Parser parser(impl_->visualizationPlant);
  configurePackageMap(&parser);
  const auto models = parser.AddModels(urdfPath);
  if (models.size() != 1) {
    throw std::runtime_error(
        "Expected exactly one visualization model in URDF: " + urdfPath);
  }
  impl_->visualizationModelInstance = models.front();

  impl_->visualizationPlant->RegisterVisualGeometry(
      impl_->visualizationPlant->world_body(),
      drake::math::RigidTransformd(Eigen::Vector3d(0.0, 0.0, -0.1)),
      geometry::Box(20.0, 20.0, 0.2), "ground_visual",
      Eigen::Vector4d(0.5, 0.5, 0.5, 1.0));
  impl_->visualizationPlant->Finalize();

  geometry::MeshcatParams meshcatParams;
  meshcatParams.host = config_.meshcatHost;
  impl_->meshcat = std::make_shared<geometry::Meshcat>(meshcatParams);
  impl_->meshcat->SetCameraPose(Eigen::Vector3d(3.0, -3.0, 2.0),
                                Eigen::Vector3d(0.0, 0.0, 0.8));

  geometry::MeshcatVisualizerParams visualizerParams;
  // Use an absolute path. A relative "drake" prefix would be appended to
  // Meshcat's own /drake namespace and produce the confusing /drake/drake
  // tree.
  visualizerParams.prefix = "/robot";
  geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, plantAndSceneGraph.scene_graph, impl_->meshcat,
      visualizerParams);

  impl_->visualizationRobotRootPath =
      "/robot/" + impl_->visualizationPlant->GetModelInstanceName(
                       impl_->visualizationModelInstance) +
      "/" + config_.rootBodyName;

  impl_->visualizationDiagram = builder.Build();
  impl_->visualizationContext =
      impl_->visualizationDiagram->CreateDefaultContext();

  if (config_.verbose) {
    std::cerr << "Meshcat visualization: " << impl_->meshcat->web_url()
              << '\n';
  }
}

void DrakeSimInterface::publishVisualization() {
  if (!impl_->visualizationDiagram) {
    return;
  }

  model::RobotState state(getRobotDescription(), 2);
  threadSafeRobotState_.copy_value(state);

  std::lock_guard<std::mutex> lock(visualizationMutex_);
  auto& plantContext =
      impl_->visualizationDiagram->GetMutableSubsystemContext(
          *impl_->visualizationPlant, impl_->visualizationContext.get());
  const auto& rootBody = impl_->visualizationPlant->GetBodyByName(
      config_.rootBodyName, impl_->visualizationModelInstance);
  const quaternion_t orientation =
      state.getRootRotationLocalToWorldFrame();
  impl_->visualizationPlant->SetFreeBodyPose(
      &plantContext, rootBody,
      drake::math::RigidTransformd(
          drake::math::RotationMatrixd(orientation.toRotationMatrix()),
          state.getRootPositionInWorldFrame()));

  for (const std::string& jointName : getRobotDescription().getJointNames()) {
    const auto jointIndex = getRobotDescription().getJointIndex(jointName);
    const auto& joint =
        impl_->visualizationPlant->GetJointByName<mbd::RevoluteJoint>(
            jointName, impl_->visualizationModelInstance);
    joint.set_angle(&plantContext, state.getJointPosition(jointIndex));
  }
  impl_->visualizationDiagram->ForcedPublish(*impl_->visualizationContext);
}

std::string DrakeSimInterface::getVisualizationUrl() const {
  return impl_->meshcat ? impl_->meshcat->web_url() : std::string();
}

std::string DrakeSimInterface::getVisualizationStatus() const {
  if (!impl_->meshcat) {
    return "disabled";
  }
  std::ostringstream status;
  status << "scene=" << impl_->meshcat->HasPath("/robot")
         << " ground=" << impl_->meshcat->HasPath("/robot/ground_visual")
         << " g1_root="
         << impl_->meshcat->HasPath(impl_->visualizationRobotRootPath)
         << " clients=" << impl_->meshcat->GetNumActiveConnections();
  return status.str();
}

uint64_t DrakeSimInterface::getFallResetCount() const {
  return impl_->fallResetCount.load();
}

void DrakeSimInterface::setupJointIndexMaps() {
  const int actuatorCount = impl_->plant->num_actuators(impl_->modelInstance);
  impl_->actuatorJointNames.reserve(actuatorCount);
  impl_->robotActuatorIndices.reserve(actuatorCount);
  impl_->effortLimits.reserve(actuatorCount);

  for (mbd::JointActuatorIndex actuatorIndex(0);
       actuatorIndex < impl_->plant->num_actuators(); ++actuatorIndex) {
    const auto& actuator = impl_->plant->get_joint_actuator(actuatorIndex);
    if (actuator.model_instance() != impl_->modelInstance) {
      continue;
    }
    const std::string jointName = actuator.joint().name();
    if (!getRobotDescription().containsJoint(jointName)) {
      throw std::runtime_error("Drake actuator is absent from RobotDescription: " +
                               jointName);
    }
    impl_->actuatorJointNames.push_back(jointName);
    impl_->robotActuatorIndices.push_back(
        getRobotDescription().getJointIndex(jointName));
    impl_->effortLimits.push_back(
        getRobotDescription().getJointDescription(jointName).max_effort);
  }

  if (impl_->actuatorJointNames.size() != getRobotDescription().getNumJoints()) {
    throw std::runtime_error(
        "Drake URDF actuator count does not match RobotDescription joint count");
  }
  impl_->actuation = Eigen::VectorXd::Zero(actuatorCount);

  if (config_.verbose) {
    std::cerr << "Drake URDF backend: nq=" << impl_->plant->num_positions()
              << " nv=" << impl_->plant->num_velocities()
              << " nu=" << actuatorCount << " dt=" << config_.dt << '\n';
    for (size_t i = 0; i < impl_->actuatorJointNames.size(); ++i) {
      std::cerr << "  actuator[" << i << "] = "
                << impl_->actuatorJointNames[i] << '\n';
    }
  }
}

void DrakeSimInterface::setSimState(const model::RobotState& robotState) {
  auto& plantContext = impl_->diagram->GetMutableSubsystemContext(
      *impl_->plant, &impl_->simulator->get_mutable_context());
  const auto& rootBody = impl_->plant->GetBodyByName(
      config_.rootBodyName, impl_->modelInstance);

  const quaternion_t rootOrientation =
      robotState.getRootRotationLocalToWorldFrame();
  const vector3_t rootPosition = robotState.getRootPositionInWorldFrame();
  impl_->plant->SetFreeBodyPose(
      &plantContext, rootBody,
      drake::math::RigidTransformd(
          drake::math::RotationMatrixd(rootOrientation.toRotationMatrix()),
          rootPosition));

  const vector3_t angularVelocityWorld =
      rootOrientation * robotState.getRootAngularVelocityInLocalFrame();
  const vector3_t linearVelocityWorld =
      rootOrientation * robotState.getRootLinearVelocityInLocalFrame();
  impl_->plant->SetFreeBodySpatialVelocity(
      &plantContext, rootBody,
      mbd::SpatialVelocity<double>(angularVelocityWorld, linearVelocityWorld));

  for (const std::string& jointName : getRobotDescription().getJointNames()) {
    const auto jointIndex = getRobotDescription().getJointIndex(jointName);
    const auto& joint = impl_->plant->GetJointByName<mbd::RevoluteJoint>(
        jointName, impl_->modelInstance);
    joint.set_angle(&plantContext, robotState.getJointPosition(jointIndex));
    joint.set_angular_rate(&plantContext,
                           robotState.getJointVelocity(jointIndex));
  }
}

void DrakeSimInterface::updateThreadSafeRobotState() {
  const auto& plantContext = impl_->diagram->GetSubsystemContext(
      *impl_->plant, impl_->simulator->get_context());
  const auto& rootBody = impl_->plant->GetBodyByName(
      config_.rootBodyName, impl_->modelInstance);
  const auto rootPose = impl_->plant->EvalBodyPoseInWorld(plantContext, rootBody);
  const auto rootVelocity =
      impl_->plant->EvalBodySpatialVelocityInWorld(plantContext, rootBody);
  const quaternion_t rootOrientation(rootPose.rotation().matrix());

  robotStateInternal_.setRootPositionInWorldFrame(rootPose.translation());
  robotStateInternal_.setRootRotationLocalToWorldFrame(rootOrientation);
  robotStateInternal_.setRootLinearVelocityInLocalFrame(
      rootOrientation.inverse() * rootVelocity.translational());
  robotStateInternal_.setRootAngularVelocityInLocalFrame(
      rootOrientation.inverse() * rootVelocity.rotational());

  for (const std::string& jointName : getRobotDescription().getJointNames()) {
    const auto jointIndex = getRobotDescription().getJointIndex(jointName);
    const auto& joint = impl_->plant->GetJointByName(
        jointName, impl_->modelInstance);
    robotStateInternal_.setJointPosition(jointIndex,
                                         joint.GetOnePosition(plantContext));
    robotStateInternal_.setJointVelocity(jointIndex,
                                         joint.GetOneVelocity(plantContext));
  }

  bool leftContact = config_.forceContactFlags;
  bool rightContact = config_.forceContactFlags;
  if (!config_.forceContactFlags) {
    const auto& contactResults =
        impl_->plant->get_contact_results_output_port()
            .Eval<mbd::ContactResults<double>>(plantContext);
    for (int i = 0; i < contactResults.num_point_pair_contacts(); ++i) {
      const auto& contact = contactResults.point_pair_contact_info(i);
      const std::string& bodyA =
          impl_->plant->get_body(contact.bodyA_index()).name();
      const std::string& bodyB =
          impl_->plant->get_body(contact.bodyB_index()).name();
      leftContact = leftContact || bodyA == config_.leftFootBodyName ||
                    bodyB == config_.leftFootBodyName;
      rightContact = rightContact || bodyA == config_.rightFootBodyName ||
                     bodyB == config_.rightFootBodyName;
    }
  }
  robotStateInternal_.setContactFlag(0, leftContact);
  robotStateInternal_.setContactFlag(1, rightContact);
  robotStateInternal_.setTime(impl_->simulator->get_context().get_time());
  threadSafeRobotState_.set(robotStateInternal_);
}

void DrakeSimInterface::resetUnlocked() {
  setSimState(initialState_);
  impl_->actuation.setZero();
  for (size_t i = 0; i < getRobotDescription().getNumJoints(); ++i) {
    auto& action = robotJointActionInternal_.at(i).value();
    action.feed_forward_effort = 0.0;
    action.q_des = initialState_.getJointPosition(i);
    action.qd_des = 0.0;
  }
  threadSafeRobotJointAction_.set(robotJointActionInternal_);

  auto& plantContext = impl_->diagram->GetMutableSubsystemContext(
      *impl_->plant, &impl_->simulator->get_mutable_context());
  impl_->plant->get_actuation_input_port(impl_->modelInstance)
      .FixValue(&plantContext, impl_->actuation);
  impl_->simulator->Initialize();
  updateThreadSafeRobotState();
}

void DrakeSimInterface::reset() {
  std::lock_guard<std::mutex> lock(drakeMutex_);
  resetUnlocked();
}

void DrakeSimInterface::simulationStep() {
  bool resetPerformed = false;
  {
    std::lock_guard<std::mutex> lock(drakeMutex_);
    threadSafeRobotJointAction_.copy_value(robotJointActionInternal_);

    const auto& plantContext = impl_->diagram->GetSubsystemContext(
        *impl_->plant, impl_->simulator->get_context());
    for (size_t i = 0; i < impl_->robotActuatorIndices.size(); ++i) {
      const joint_index_t robotIndex = impl_->robotActuatorIndices[i];
      const auto& action = robotJointActionInternal_.at(robotIndex).value();
      const auto& joint = impl_->plant->GetJointByName(
          impl_->actuatorJointNames[i], impl_->modelInstance);
      double torque = action.getTotalFeedbackTorque(
          joint.GetOnePosition(plantContext),
          joint.GetOneVelocity(plantContext));
      const double effortLimit = impl_->effortLimits[i];
      if (!std::isfinite(torque)) {
        torque = 0.0;
      } else {
        torque = std::clamp(torque, -effortLimit, effortLimit);
      }
      impl_->actuation[static_cast<Eigen::Index>(i)] = torque;
    }

    auto& mutablePlantContext = impl_->diagram->GetMutableSubsystemContext(
        *impl_->plant, &impl_->simulator->get_mutable_context());
    impl_->plant->get_actuation_input_port(impl_->modelInstance)
        .FixValue(&mutablePlantContext, impl_->actuation);

    const double nextTime =
        impl_->simulator->get_context().get_time() + config_.dt;
    impl_->simulator->AdvanceTo(nextTime);
    updateThreadSafeRobotState();

    if (config_.resetOnFall &&
        robotStateInternal_.getRootPositionInWorldFrame().z() <
            config_.fallHeight) {
      impl_->fallResetCount.fetch_add(1);
      resetUnlocked();
      resetPerformed = true;
    }
  }

  if (resetPerformed && config_.resetPause > 0.0) {
    std::this_thread::sleep_for(
        std::chrono::duration<double>(config_.resetPause));
  }
}

void DrakeSimInterface::simulationLoop() {
  const auto stepDuration = std::chrono::microseconds(
      std::max<int64_t>(1, static_cast<int64_t>(std::llround(config_.dt * 1e6))));
  auto nextWakeup = std::chrono::steady_clock::now();

  while (!terminate_.load()) {
    try {
      simulationStep();
    } catch (const std::exception& error) {
      std::cerr << "Drake simulation step failed: " << error.what() << '\n';
      std::lock_guard<std::mutex> lock(drakeMutex_);
      impl_->fallResetCount.fetch_add(1);
      resetUnlocked();
    }
    nextWakeup += stepDuration;
    std::this_thread::sleep_until(nextWakeup);
    if (std::chrono::steady_clock::now() > nextWakeup + stepDuration) {
      nextWakeup = std::chrono::steady_clock::now();
    }
  }
}

void DrakeSimInterface::initSim() {
  if (simInitialized_) {
    return;
  }
  simulationStep();
  simInitialized_ = true;
}

void DrakeSimInterface::startSim() {
  if (simStarted_) {
    return;
  }
  if (!simInitialized_) {
    initSim();
  }
  simStarted_ = true;
  simulationThread_ =
      std::thread(&DrakeSimInterface::simulationLoop, this);
}

}  // namespace robot::drake_sim_interface
