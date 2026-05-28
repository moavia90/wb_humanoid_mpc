// Standalone Drake SAP PD standing test — no MPC, no OCS2, no ROS 2 node.
//
// Loads g1_29dof_drake.xml, registers PD gains on every actuator via Drake's
// built-in SapPdControllerConstraint (desired_state_input_port), and runs
// the simulation for 10 seconds.
//
// Purpose: verify that Drake physics + SAP PD alone can hold the robot
// upright before blaming the MPC for instability.
//
// Open Meshcat at http://localhost:7001 to watch.
// Exit code 0 = robot stayed upright. Exit code 1 = robot fell.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <drake/geometry/meshcat.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/proximity_properties.h>
#include <drake/geometry/scene_graph.h>
#include <drake/math/rigid_transform.h>
#include <drake/math/rotation_matrix.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/joint_actuator.h>
#include <drake/multibody/tree/revolute_joint.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram_builder.h>

namespace drake_mbd = drake::multibody;
namespace drake_geo = drake::geometry;
namespace drake_math = drake::math;
namespace drake_sys = drake::systems;

// --------------------------------------------------------------------------
// Nominal standing pose: moderate knee bend for a stable standing posture.
// --------------------------------------------------------------------------
static double nominalAngle(const std::string& joint_name) {
  if (joint_name.find("hip_pitch") != std::string::npos) return -0.20;
  if (joint_name.find("knee") != std::string::npos) return 0.40;
  if (joint_name.find("ankle_pitch") != std::string::npos) return -0.20;
  return 0.0;
}

// --------------------------------------------------------------------------
// Per-joint PD gains.
// --------------------------------------------------------------------------
static std::pair<double, double> pdGains(const std::string& joint_name) {
  //                                           Kp      Kd
  if (joint_name.find("hip_pitch") != std::string::npos) return {300.0, 10.0};
  if (joint_name.find("hip_roll") != std::string::npos) return {200.0, 6.0};
  if (joint_name.find("hip_yaw") != std::string::npos) return {100.0, 4.0};
  if (joint_name.find("knee") != std::string::npos) return {400.0, 12.0};
  if (joint_name.find("ankle_pitch") != std::string::npos) return {150.0, 6.0};
  if (joint_name.find("ankle_roll") != std::string::npos) return {80.0, 3.0};
  if (joint_name.find("waist") != std::string::npos) return {200.0, 5.0};
  if (joint_name.find("shoulder") != std::string::npos) return {50.0, 2.0};
  if (joint_name.find("elbow") != std::string::npos) return {50.0, 2.0};
  if (joint_name.find("wrist") != std::string::npos) return {20.0, 1.0};
  return {50.0, 2.0};
}

int main() {
  constexpr double kSimDt = 0.0005;        // match drake_mrt_sim_node
  constexpr double kTestDuration = 10.0;   // seconds to stand
  constexpr double kFallThreshold = 0.50;  // pelvis_z below this = fallen
  constexpr double kFallAngleThresholdRad = 0.90;
  constexpr double kReportInterval = 0.05;  // console update every 50 ms

  // ── Locate model ────────────────────────────────────────────────────────
  std::string model_path;
  try {
    model_path = ament_index_cpp::get_package_share_directory("g1_description") + "/urdf/g1_29dof_drake.xml";
  } catch (const std::exception& e) {
    std::cerr << "[sap_stand_test] Cannot find g1_description: " << e.what() << "\n"
              << "  Make sure the workspace is sourced.\n";
    return 1;
  }
  std::cout << "[sap_stand_test] Model : " << model_path << "\n";

  // ── Build diagram ────────────────────────────────────────────────────────
  drake_sys::DiagramBuilder<double> builder;
  auto [plant, scene_graph] = drake_mbd::AddMultibodyPlantSceneGraph(&builder, kSimDt);

  // kPointContactOnly: the simplest and most predictable discrete contact model.
  // No hydroelastic fallback complications — foot spheres and ground use the
  // same point-contact stiffness path through the SAP solver.
  plant.set_contact_model(drake_mbd::ContactModel::kPointContactOnly);
  plant.set_discrete_contact_approximation(drake_mbd::DiscreteContactApproximation::kSap);

  // Parse G1 model
  drake_mbd::Parser parser(&plant);
  parser.package_map().PopulateFromEnvironment("AMENT_PREFIX_PATH");
  try {
    const std::string share = ament_index_cpp::get_package_share_directory("g1_description");
    if (!parser.package_map().Contains("g1_description")) parser.package_map().Add("g1_description", share);
  } catch (...) {
  }
  const auto model_instance = parser.AddModels(model_path).at(0);

  // Flat ground: point-contact only.
  // Stiffness 5e4 N/m: equilibrium penetration ≈ 0.6 mm per 8-sphere foot set.
  // k_eff = k_ground * k_sphere_default / (k_ground + k_sphere_default)
  //       = 5e4 * 1e6 / (5e4 + 1e6) ≈ 4.76e4 N/m (sphere default = 1e6)
  {
    drake_geo::ProximityProperties ground_props;
    drake_geo::AddContactMaterial(2.0,    // match drake_mrt_sim_node
                                  5.0e4,  // point_contact_stiffness
                                  drake_mbd::CoulombFriction<double>(0.9, 0.7), &ground_props);
    plant.RegisterCollisionGeometry(plant.world_body(), drake_math::RigidTransformd(Eigen::Vector3d(0.0, 0.0, -0.1)),
                                    drake_geo::Box(20.0, 20.0, 0.2), "ground", ground_props);
    plant.RegisterVisualGeometry(plant.world_body(), drake_math::RigidTransformd(Eigen::Vector3d(0.0, 0.0, -0.2)),
                                 drake_geo::Box(20.0, 20.0, 0.2), "ground_visual", Eigen::Vector4d(0.5, 0.5, 0.5, 1.0));
  }

  // ── Register SAP PD gains BEFORE Finalize ───────────────────────────────
  const int n_act = plant.num_actuators();
  std::cout << "[sap_stand_test] Setting SAP PD gains on " << n_act << " actuators:\n";
  for (drake_mbd::JointActuatorIndex i(0); i < n_act; ++i) {
    auto& actuator = plant.get_mutable_joint_actuator(i);
    const auto [Kp, Kd] = pdGains(actuator.joint().name());
    actuator.set_controller_gains({Kp, Kd});
    std::printf("  %-35s  Kp=%6.1f  Kd=%5.1f\n", actuator.joint().name().c_str(), Kp, Kd);
  }

  plant.Finalize();

  // ── Meshcat ──────────────────────────────────────────────────────────────
  drake_geo::MeshcatParams meshcat_params;
  meshcat_params.host = "*";
  meshcat_params.port = 7001;
  auto meshcat = std::make_shared<drake_geo::Meshcat>(meshcat_params);

  drake_geo::MeshcatVisualizerParams viz_params;
  viz_params.publish_period = 1.0 / 60.0;
  drake_geo::MeshcatVisualizer<double>::AddToBuilder(&builder, scene_graph, meshcat, viz_params);

  auto diagram = builder.Build();
  auto sim = std::make_unique<drake_sys::Simulator<double>>(*diagram);
  sim->set_target_realtime_rate(0.0);  // run as fast as possible

  auto& context = sim->get_mutable_context();
  auto& plant_context = diagram->GetMutableSubsystemContext(plant, &context);

  // ── Compute spawn height from foot positions ─────────────────────────────
  constexpr double kRefHeight = 1.0;
  plant.SetFreeBodyPose(&plant_context, plant.GetBodyByName("pelvis"), drake_math::RigidTransformd(Eigen::Vector3d(0.0, 0.0, kRefHeight)));
  {
    drake_mbd::SpatialVelocity<double> zero;
    zero.SetZero();
    plant.SetFreeBodySpatialVelocity(&plant_context, plant.GetBodyByName("pelvis"), zero);
  }
  for (drake_mbd::JointActuatorIndex i(0); i < n_act; ++i) {
    try {
      const auto& joint = dynamic_cast<const drake_mbd::RevoluteJoint<double>&>(plant.get_joint_actuator(i).joint());
      joint.set_angle(&plant_context, nominalAngle(joint.name()));
      joint.set_angular_rate(&plant_context, 0.0);
    } catch (...) {
    }
  }

  // Find the lowest foot contact sphere center (spheres are at z=-0.03
  // relative to ankle_roll_link body frame in the nominal standing pose).
  double lowest_foot_z = kRefHeight;
  for (const char* link : {"left_ankle_roll_link", "right_ankle_roll_link"}) {
    try {
      const auto pose = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName(link));
      lowest_foot_z = std::min(lowest_foot_z, pose.translation().z() - 0.03);
    } catch (...) {
    }
  }

  // Sphere radius = 0.005 m.
  // Place sphere centres at z = 0.005 (bottoms exactly at ground, z = 0).
  // SAP implicit solver computes the correct contact forces even from zero
  // penetration — it anticipates the closing motion and applies preemptive
  // normal forces to satisfy the no-penetration constraint at the next step.
  constexpr double kSphereCentreTarget = 0.005;  // z of foot sphere centres
  const double spawn_height = (kRefHeight - lowest_foot_z) + kSphereCentreTarget;
  std::cout << "[sap_stand_test] Spawn height: " << spawn_height << " m  " << "(sphere centres at z=" << kSphereCentreTarget << ", "
            << "initial penetration=" << (0.005 - kSphereCentreTarget) * 1000 << " mm)\n";

  // Apply final initial pose
  plant.SetFreeBodyPose(&plant_context, plant.GetBodyByName("pelvis"),
                        drake_math::RigidTransformd(Eigen::Vector3d(0.0, 0.0, spawn_height)));

  // ── Desired state: nominal standing, zero velocity ───────────────────────
  Eigen::VectorXd desired_state = Eigen::VectorXd::Zero(2 * n_act);
  for (drake_mbd::JointActuatorIndex i(0); i < n_act; ++i) {
    desired_state[static_cast<int>(i)] = nominalAngle(plant.get_joint_actuator(i).joint().name());
  }
  plant.get_desired_state_input_port(model_instance).FixValue(&plant_context, desired_state);
  plant.get_actuation_input_port(model_instance).FixValue(&plant_context, Eigen::VectorXd::Zero(n_act));

  sim->Initialize();
  diagram->ForcedPublish(context);

  std::cout << "[sap_stand_test] Running " << kTestDuration << "s. " << "Meshcat: " << meshcat->web_url() << "\n";
  std::cout << "[sap_stand_test] " << "   t(s)  pelvis_z  pitch°   roll°  status\n";
  std::cout << "              " << "-------------------------------------------\n";

  // ── Simulation loop ──────────────────────────────────────────────────────
  bool ever_fallen = false;
  double t = 0.0;
  while (t < kTestDuration) {
    const double t_next = std::min(t + kReportInterval, kTestDuration);
    sim->AdvanceTo(t_next);
    t = t_next;

    const auto& pelvis = plant.GetBodyByName("pelvis");
    const auto X = plant.EvalBodyPoseInWorld(plant_context, pelvis);
    const double pelvis_z = X.translation().z();
    const auto R = X.rotation().matrix();
    const double pitch_deg = std::asin(std::clamp(-R(2, 0), -1.0, 1.0)) * 180.0 / M_PI;
    const double roll_deg = std::atan2(R(2, 1), R(2, 2)) * 180.0 / M_PI;

    const bool standing = pelvis_z > kFallThreshold && std::abs(pitch_deg * M_PI / 180.0) < kFallAngleThresholdRad &&
                          std::abs(roll_deg * M_PI / 180.0) < kFallAngleThresholdRad;
    if (!standing) ever_fallen = true;

    std::printf("[sap_stand_test]  %5.2f  %7.4f  %6.2f  %7.2f  %s\n", t, pelvis_z, pitch_deg, roll_deg, standing ? "OK" : "*** FALLEN ***");

    if (!standing) {
      std::cout << "[sap_stand_test] Robot fell at t=" << t << "s — stopping early.\n";
      break;
    }
  }

  // ── Result ───────────────────────────────────────────────────────────────
  if (!ever_fallen) {
    std::cout << "\n[sap_stand_test] PASS: robot stood for " << kTestDuration << "s with SAP PD only.\n";
    return 0;
  } else {
    std::cout << "\n[sap_stand_test] FAIL: robot fell under pure SAP PD control.\n"
              << "  Possible causes:\n"
              << "  1. kPointContactOnly not detecting foot sphere collisions\n"
              << "  2. PD gains too low — increase hip/knee Kp in pdGains()\n"
              << "  3. Nominal pose not in static equilibrium — adjust nominalAngle()\n"
              << "  4. Contact stiffness mismatch — edit ground AddContactMaterial()\n";
    return 1;
  }
}
