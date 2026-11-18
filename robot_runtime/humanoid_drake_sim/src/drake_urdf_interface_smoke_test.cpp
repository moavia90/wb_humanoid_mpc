#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include <humanoid_drake_sim/DrakeSimInterface.h>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotState.h>

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3 ||
      (argc == 3 && std::string(argv[2]) != "--meshcat")) {
    std::cerr <<
        "Usage: drake_urdf_interface_smoke_test robot.urdf [--meshcat]\n";
    return 1;
  }

  const std::string urdfPath = argv[1];
  robot::model::RobotDescription description(urdfPath);
  auto initialState =
      std::make_shared<robot::model::RobotState>(description, 2);
  initialState->setConfigurationToZero();
  initialState->setRootPositionInWorldFrame(robot::vector3_t(0.0, 0.0, 1.0));

  robot::drake_sim_interface::DrakeSimConfig config;
  config.initStatePtr = initialState;
  config.resetOnFall = false;
  config.forceContactFlags = false;
  config.verbose = true;
  config.enableVisualization = argc == 3;

  robot::drake_sim_interface::DrakeSimInterface simulation(config, urdfPath);
  for (int i = 0; i < 10; ++i) {
    simulation.simulationStep();
  }
  simulation.updateInterfaceStateFromRobot();
  if (config.enableVisualization) {
    simulation.publishVisualization();
    std::cout << "Meshcat: " << simulation.getVisualizationUrl() << '\n'
              << "Meshcat server scene: "
              << simulation.getVisualizationStatus() << '\n';
  }

  const auto& state = simulation.getRobotState();
  const double expectedTime = 10.0 * config.dt;
  const bool timeAdvanced = state.getTime() >= expectedTime - 1.0e-12;
  const bool stateIsFinite =
      state.getRootPositionInWorldFrame().allFinite() &&
      state.getRootLinearVelocityInLocalFrame().allFinite() &&
      state.getRootAngularVelocityInLocalFrame().allFinite();

  if (!timeAdvanced || !stateIsFinite) {
    std::cerr << "FAIL: invalid Drake state after ten URDF simulation steps\n";
    return 2;
  }

  std::cout << "PASS: URDF backend advanced to t=" << state.getTime()
            << " with " << simulation.getRobotDescription().getNumJoints()
            << " named joints\n";
  return 0;
}
