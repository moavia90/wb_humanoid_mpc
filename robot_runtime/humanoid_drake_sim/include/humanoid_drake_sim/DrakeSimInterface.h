#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <robot_model/RobotHWInterfaceBase.h>

namespace robot::drake_sim_interface {

/** Physics-only settings for the Drake hardware-interface backend. */
struct DrakeSimConfig {
  std::shared_ptr<model::RobotState> initStatePtr;
  double dt{0.0005};
  double jointDamping{10.0};
  double groundStaticFriction{0.9};
  double groundDynamicFriction{0.7};
  double groundDissipation{2.0};
  double groundPointStiffness{5.0e4};
  // Compatibility with the current MuJoCo backend, which reports both feet
  // in contact unconditionally. Set false to use Drake contact results.
  bool forceContactFlags{true};
  bool resetOnFall{true};
  double fallHeight{0.2};
  double resetPause{1.0};
  bool verbose{false};
  bool enableVisualization{false};
  // Drake uses "*" as the documented all-interfaces wildcard. Its web_url()
  // renders that wildcard as "localhost", which is safe to paste in a browser.
  std::string meshcatHost{"*"};
  std::string rootBodyName{"pelvis"};
  std::string leftFootBodyName{"left_ankle_roll_link"};
  std::string rightFootBodyName{"right_ankle_roll_link"};
};

/**
 * Drake implementation of the same RobotHWInterfaceBase contract used by
 * MujocoSimInterface.
 *
 * The controller writes RobotJointAction through applyJointAction(), while a
 * dedicated physics thread consumes the latest action and publishes
 * RobotState snapshots.  This class contains no standing controller, startup
 * pin, torque blending, or policy logic.
 */
class DrakeSimInterface final : public model::RobotHWInterfaceBase {
 public:
  DrakeSimInterface(const DrakeSimConfig& config, const std::string& urdfPath);
  ~DrakeSimInterface();

  DrakeSimInterface(const DrakeSimInterface&) = delete;
  DrakeSimInterface& operator=(const DrakeSimInterface&) = delete;

  void initSim();
  void startSim();
  void simulationStep();
  void reset();

  /** Publish a thread-safe state snapshot to Meshcat from the calling thread. */
  void publishVisualization();
  std::string getVisualizationUrl() const;
  /** Human-readable server-side scene check; call from the construction thread. */
  std::string getVisualizationStatus() const;
  uint64_t getFallResetCount() const;

  const DrakeSimConfig& getConfig() const { return config_; }

 private:
  struct Impl;

  void buildPlant(const std::string& urdfPath);
  void buildVisualization(const std::string& urdfPath);
  void setupJointIndexMaps();
  void setSimState(const model::RobotState& robotState);
  void resetUnlocked();
  void updateThreadSafeRobotState();
  void simulationLoop();

  DrakeSimConfig config_;
  model::RobotState robotStateInternal_;
  model::RobotJointAction robotJointActionInternal_;
  model::RobotState initialState_;
  std::unique_ptr<Impl> impl_;

  std::atomic<bool> terminate_{false};
  bool simInitialized_{false};
  bool simStarted_{false};
  std::thread simulationThread_;
  std::mutex drakeMutex_;
  std::mutex visualizationMutex_;
};

}  // namespace robot::drake_sim_interface
