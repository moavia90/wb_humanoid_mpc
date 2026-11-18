/******************************************************************************
gravity_comp_hold_test.cpp  —  STAGE 3: GRAVITY-COMP HOLD (physics on, MPC off)

Answers ONE question: "Can Drake's contact model + damping keep the G1
standing AT ALL, with the MPC entirely out of the picture?"

Control law (no MPC, no MRT):
  tau = -g_q(actuated)                       gravity compensation
        + Kp (q_nom - q) + Kd (0 - qd)       light posture PD
        - d_passive * qd                     MuJoCo-equivalent damping
  clamped to joint torque limits.

Interpretation:
  - Robot stands N seconds -> Drake physics setup (foot geometry, contact,
    damping) is sound. The MPC is now the only remaining suspect.
  - Robot tips over -> the problem is foot geometry / contact / damping.
    The MPC is exonerated; fix the model before testing any policy.

NOTE: per Stage 2 / Q1 discussion, joint PD + gravity comp does NOT regulate
COM. This test passes only if the static pose is genuinely feasible
(Stage 2 gate) and the support polygon is real. That is exactly the point.

Pure Drake — links drake::drake only, no OCS2/robot_model, no absl stub.

Usage:
  gravity_comp_hold_test <drake_model.xml> [duration_s=10] [pelvis_z=0.78]
******************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <drake/geometry/meshcat.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/proximity_properties.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/shape_specification.h>
#include <drake/math/rigid_transform.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/joint_actuator.h>
#include <drake/multibody/tree/revolute_joint.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram_builder.h>

namespace mbd = drake::multibody;
namespace geo = drake::geometry;

double nominalQ(const std::string& j) {
  if (j.find("hip_pitch") != std::string::npos) return -0.20;
  if (j.find("knee") != std::string::npos) return 0.40;
  if (j.find("ankle_pitch") != std::string::npos) return -0.20;
  return 0.0;
}

double torqueLimit(const std::string& j) {
  if (j.find("knee") != std::string::npos) return 139.0;
  if (j.find("hip_") != std::string::npos) return 88.0;
  if (j.find("ankle") != std::string::npos) return 50.0;
  if (j.find("waist_yaw") != std::string::npos) return 88.0;
  if (j.find("waist") != std::string::npos) return 50.0;
  if (j.find("wrist_pitch") != std::string::npos ||
      j.find("wrist_yaw") != std::string::npos) return 5.0;
  return 25.0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
        "Usage: gravity_comp_hold_test <drake_model.xml> [dur=10] [pelvis_z=0.78]\n");
    return 1;
  }
  const std::string model_path = argv[1];
  const double duration = (argc > 2) ? std::stod(argv[2]) : 10.0;
  const double pelvis_z = (argc > 3) ? std::stod(argv[3]) : 0.78;

  constexpr double kSimDt = 0.0005;
  constexpr double kPassiveDamping = 10.0;
  constexpr double kSettleTime = 0.5;   // brief pin so contact engages cleanly

  /* --------------------------- Build plant ----------------------------- */
  drake::systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] = mbd::AddMultibodyPlantSceneGraph(&builder, kSimDt);
  plant.set_contact_model(mbd::ContactModel::kPointContactOnly);
  plant.set_discrete_contact_approximation(mbd::DiscreteContactApproximation::kSap);

  mbd::Parser parser(&plant);
  const auto mi = parser.AddModels(model_path).at(0);

  geo::ProximityProperties gp;
  geo::AddContactMaterial(2.0, 5e4, mbd::CoulombFriction<double>(1.2, 1.0), &gp);
  plant.RegisterCollisionGeometry(plant.world_body(),
      drake::math::RigidTransformd(Eigen::Vector3d(0, 0, -0.1)),
      geo::Box(20, 20, 0.2), "ground_collision", gp);
  plant.RegisterVisualGeometry(plant.world_body(),
      drake::math::RigidTransformd(Eigen::Vector3d(0, 0, -0.1)),
      geo::Box(20, 20, 0.2), "ground_visual",
      Eigen::Vector4d(0.5, 0.5, 0.5, 1.0));

  plant.Finalize();

  auto meshcat = std::make_shared<geo::Meshcat>();
  geo::MeshcatVisualizer<double>::AddToBuilder(&builder, scene_graph, meshcat);
  std::printf("Meshcat: %s\n", meshcat->web_url().c_str());

  auto diagram = builder.Build();
  drake::systems::Simulator<double> sim(*diagram);
  auto& plant_ctx =
      diagram->GetMutableSubsystemContext(plant, &sim.get_mutable_context());

  const int n_act = plant.num_actuators();
  std::vector<std::string> act_names;
  std::vector<int> act_v_index;  // generalized-velocity index per actuator
  for (mbd::JointActuatorIndex i(0); i < n_act; ++i) {
    const auto& joint = plant.get_joint_actuator(i).joint();
    act_names.push_back(joint.name());
    act_v_index.push_back(joint.velocity_start());
  }

  /* ------------------------ Spawn at nominal --------------------------- */
  const auto& pelvis = plant.GetBodyByName("pelvis");
  plant.SetFreeBodyPose(&plant_ctx, pelvis,
      drake::math::RigidTransformd(drake::math::RotationMatrixd::Identity(),
                                   Eigen::Vector3d(0, 0, pelvis_z)));
  for (const auto& jn : act_names) {
    try {
      const auto& j = dynamic_cast<const mbd::RevoluteJoint<double>&>(
          plant.GetJointByName(jn));
      j.set_angle(&plant_ctx, nominalQ(jn));
      j.set_angular_rate(&plant_ctx, 0.0);
    } catch (...) {}
  }
  sim.Initialize();

  /* ------------------------ Hold-and-watch loop ------------------------ */
  std::printf("\n%8s %10s %10s %10s %10s %10s\n",
              "t", "pelvis_z", "pitch_deg", "roll_deg", "tau_inf", "status");

  Eigen::VectorXd tau(n_act);
  double t = 0.0;
  bool fell = false;
  int step = 0;
  const int log_every = static_cast<int>(0.25 / kSimDt);

  while (t < duration && !fell) {
    // Gravity compensation: u = -tau_g at the actuated DOFs.
    const Eigen::VectorXd tau_g = plant.CalcGravityGeneralizedForces(plant_ctx);

    for (int a = 0; a < n_act; ++a) {
      const std::string& jn = act_names[a];
      double q = 0, qd = 0;
      try {
        const auto& j = dynamic_cast<const mbd::RevoluteJoint<double>&>(
            plant.GetJointByName(jn));
        q = j.get_angle(plant_ctx);
        qd = j.get_angular_rate(plant_ctx);
      } catch (...) {}

      // Light posture PD only — deliberately weak; the gravity comp does
      // the heavy lifting, and we are NOT trying to hide a bad support
      // polygon behind stiff gains.
      double kp = 60, kd = 4;
      if (jn.find("knee") != std::string::npos) { kp = 100; kd = 6; }
      if (jn.find("hip_pitch") != std::string::npos) { kp = 100; kd = 6; }
      if (jn.find("ankle") != std::string::npos) { kp = 40; kd = 3; }

      double ti = -tau_g[act_v_index[a]]
                  + kp * (nominalQ(jn) - q) - kd * qd
                  - kPassiveDamping * qd;

      const double lim = torqueLimit(jn);
      tau[a] = std::clamp(ti, -lim, lim);
    }

    // Brief settle pin: hold base pose for the first kSettleTime so the
    // contact engages from rest rather than from a drop.
    if (t < kSettleTime) {
      plant.SetFreeBodyPose(&plant_ctx, pelvis,
          drake::math::RigidTransformd(drake::math::RotationMatrixd::Identity(),
                                       Eigen::Vector3d(0, 0, pelvis_z)));
      mbd::SpatialVelocity<double> zero; zero.SetZero();
      plant.SetFreeBodySpatialVelocity(&plant_ctx, pelvis, zero);
    }

    plant.get_actuation_input_port(mi).FixValue(&plant_ctx, tau);

    t += kSimDt;
    sim.AdvanceTo(t);

    const auto X = plant.EvalBodyPoseInWorld(plant_ctx, pelvis);
    const auto R = X.rotation().matrix();
    const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
    const double roll = std::atan2(R(2, 1), R(2, 2));
    const double pz = X.translation().z();
    fell = (pz < 0.35) || (std::abs(pitch) > 0.9) || (std::abs(roll) > 0.9);

    if ((step++ % log_every) == 0 || fell) {
      std::printf("%8.3f %10.4f %10.2f %10.2f %10.1f %10s\n",
                  t, pz, pitch * 57.2958, roll * 57.2958,
                  tau.cwiseAbs().maxCoeff(), fell ? "FELL" : "ok");
    }
  }

  std::printf("\n================ GRAVITY-COMP HOLD VERDICT ================\n");
  if (fell) {
    std::printf("Fell at t=%.2f s with NO MPC in the loop.\n"
                "-> Foot geometry / contact / damping problem. The MPC is\n"
                "   exonerated. Fix the model (sole geometry, Stage 2 gate)\n"
                "   before testing any policy.\n", t);
    return 2;
  }
  std::printf("Stood for %.1f s with gravity comp + light PD only.\n"
              "-> Drake physics setup is sound. Proceed to Stage 4\n"
              "   (policy_playback_test): any fall there is the policy.\n",
              duration);
  return 0;
}
