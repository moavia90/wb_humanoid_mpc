/******************************************************************************
policy_playback_test.cpp  —  STAGE 4: OPEN-LOOP MPC POLICY PLAYBACK
(rev 2: solve via SqpMpc — the API this fork actually exposes — instead of
 calling SqpSolver::run directly; const-correct getJointVelocities)

  1. Build the OCS2 MPC exactly as drake_mrt_sim_node does (SqpMpc).
  2. Set a stand-in-place target (zero commanded velocity) on the reference
     manager, run the MPC ONCE from the nominal state, extract the primal.
  3. Spawn the robot in Drake at the same state; play the plan back
     open-loop with PD-to-plan + damping + clamps.
  4. Log divergence and give a verdict.

Usage (same first 6 args as drake_mrt_sim_node):
  policy_playback_test g1 task.info reference.info g1_29dof.urdf gait.info \
                       g1_29dof_drake.xml [playback_duration_s=3.0]
******************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <drake/geometry/proximity_properties.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/shape_specification.h>
#include <drake/math/rigid_transform.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/revolute_joint.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram_builder.h>

#include <ocs2_core/misc/LinearInterpolation.h>
#include <ocs2_sqp/SqpMpc.h>

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h>

namespace mbd = drake::multibody;
namespace geo = drake::geometry;
using namespace ocs2;
using namespace ocs2::humanoid;

double nominalQ(const std::string& j) {
  if (j.find("hip_pitch") != std::string::npos) return -0.20;
  if (j.find("knee") != std::string::npos) return 0.40;
  if (j.find("ankle_pitch") != std::string::npos) return -0.20;
  return 0.0;
}

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr,
        "Usage: policy_playback_test robot task ref urdf gait drake_model [dur_s]\n");
    return 1;
  }
  const std::string task_file = argv[2];
  const std::string reference_file = argv[3];
  const std::string urdf_file = argv[4];
  const std::string drake_model = argv[6];
  const double playback_duration = (argc > 7) ? std::stod(argv[7]) : 3.0;

  constexpr double kSimDt = 0.0005;
  constexpr double kPassiveDamping = 10.0;
  constexpr double kMaxTorque = 180.0;

  /* ======================= 1. SOLVE THE MPC ONCE ======================== */

  CentroidalMpcInterface interface(task_file, urdf_file, reference_file);
  const auto& mpc_model = interface.getMpcRobotModel();
  const auto& joint_names = interface.modelSettings().mpcModelJointNames;

  const vector_t x0 = interface.getInitialState();
  const scalar_t horizon = interface.mpcSettings().timeHorizon_;

  // Stand-in-place target: zero commanded velocity from x0 (same path the
  // main node's motion manager uses, with v_cmd = 0).
  CentroidalMpcTargetTrajectoriesCalculator target_calc(
      reference_file, interface.getMpcRobotModel(), interface.getPinocchioInterface(),
      interface.getCentroidalModelInfo(), horizon);
  auto target = target_calc.commandedVelocityToTargetTrajectories(
      vector4_t::Zero(), 0.0, x0);
  interface.getReferenceManagerPtr()->setTargetTrajectories(std::move(target));

  // Same MPC construction as drake_mrt_sim_node.
  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(),
             interface.getOptimalControlProblem(), interface.getInitializer());
  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());

  std::printf("Solving MPC once: horizon=%.2fs, stateDim=%ld\n",
              horizon, static_cast<long>(x0.size()));
  mpc.run(0.0, x0);

  PrimalSolution primal;
  mpc.getSolverPtr()->getPrimalSolution(horizon, &primal);

  std::printf("Primal solution: %zu knots, t=[%.3f, %.3f]\n",
              primal.timeTrajectory_.size(),
              primal.timeTrajectory_.front(), primal.timeTrajectory_.back());

  // Sanity: contact Fz at t=0 — catches the 2x normalization bug up front.
  {
    const vector_t u0 = primal.inputTrajectory_.front();
    const double fz_l = u0[2], fz_r = u0[8];
    std::printf("t=0 contact Fz: L=%.1f N, R=%.1f N (expect ~172 each for 35 kg)\n",
                fz_l, fz_r);
    if (fz_l + fz_r > 1.6 * 35.0 * 9.81) {
      std::printf("*** WARNING: total Fz >> m*g. Fix MPC normalization (Stage 0)\n"
                  "*** before trusting this playback.\n");
    }
  }

  /* ==================== 2. BUILD DRAKE, SPAWN AT x0 ===================== */

  drake::systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] = mbd::AddMultibodyPlantSceneGraph(&builder, kSimDt);
  plant.set_contact_model(mbd::ContactModel::kPointContactOnly);
  plant.set_discrete_contact_approximation(mbd::DiscreteContactApproximation::kSap);

  mbd::Parser parser(&plant);
  const auto model_instance = parser.AddModels(drake_model).at(0);

  geo::ProximityProperties gp;
  geo::AddContactMaterial(2.0, 5e4, mbd::CoulombFriction<double>(0.9, 0.7), &gp);
  plant.RegisterCollisionGeometry(plant.world_body(),
      drake::math::RigidTransformd(Eigen::Vector3d(0, 0, -0.1)),
      geo::Box(20, 20, 0.2), "ground_collision", gp);

  plant.Finalize();
  auto diagram = builder.Build();
  drake::systems::Simulator<double> sim(*diagram);
  auto& plant_ctx = diagram->GetMutableSubsystemContext(plant, &sim.get_mutable_context());

  const int n_act = plant.num_actuators();
  std::vector<std::string> act_names;
  for (mbd::JointActuatorIndex i(0); i < n_act; ++i)
    act_names.push_back(plant.get_joint_actuator(i).joint().name());

  // Map MPC joint order -> Drake actuator index.
  std::vector<int> mpc_to_act(joint_names.size(), -1);
  for (size_t m = 0; m < joint_names.size(); ++m)
    for (int a = 0; a < n_act; ++a)
      if (act_names[a] == joint_names[m]) { mpc_to_act[m] = a; break; }

  // Spawn at the MPC's initial state: base pose + joint angles from x0.
  const auto& pelvis = plant.GetBodyByName("pelvis");
  const vector_t base_pos = mpc_model.getBasePosition(x0);
  plant.SetFreeBodyPose(&plant_ctx, pelvis,
      drake::math::RigidTransformd(drake::math::RotationMatrixd::Identity(),
          Eigen::Vector3d(base_pos[0], base_pos[1], base_pos[2])));

  const vector_t q_mpc0 = mpc_model.getJointAngles(x0);
  for (size_t m = 0; m < joint_names.size(); ++m) {
    try {
      const auto& j = dynamic_cast<const mbd::RevoluteJoint<double>&>(
          plant.GetJointByName(joint_names[m]));
      j.set_angle(&plant_ctx, q_mpc0[static_cast<Eigen::Index>(m)]);
      j.set_angular_rate(&plant_ctx, 0.0);
    } catch (...) {}
  }
  // Non-MPC joints (wrists): nominal.
  for (const auto& an : act_names) {
    bool in_mpc = false;
    for (const auto& jn : joint_names) if (jn == an) { in_mpc = true; break; }
    if (in_mpc) continue;
    try {
      const auto& j = dynamic_cast<const mbd::RevoluteJoint<double>&>(
          plant.GetJointByName(an));
      j.set_angle(&plant_ctx, nominalQ(an));
      j.set_angular_rate(&plant_ctx, 0.0);
    } catch (...) {}
  }

  sim.Initialize();

  /* ============ 3. PLAY BACK PLANNED TORQUES, LOG DIVERGENCE ============ */

  std::printf("\n%8s %10s %10s %10s %10s %10s\n",
              "t", "pelvis_z", "pitch_deg", "q_err_rms", "tau_inf", "status");

  Eigen::VectorXd tau(n_act);
  double t = 0.0;
  bool fell = false;

  const double t_end = std::min(playback_duration,
                                primal.timeTrajectory_.back() - 1e-3);
  int step = 0;
  const int log_every = static_cast<int>(0.1 / kSimDt);  // 10 Hz log

  while (t < t_end && !fell) {
    const vector_t x_plan = LinearInterpolation::interpolate(
        t, primal.timeTrajectory_, primal.stateTrajectory_);
    const vector_t u_plan = LinearInterpolation::interpolate(
        t, primal.timeTrajectory_, primal.inputTrajectory_);

    // PD-to-plan + damping (what the MRT joint controller produces from
    // (x_plan, u_plan) with feedforward disabled).
    const vector_t q_plan = mpc_model.getJointAngles(x_plan);
    const vector_t qd_plan = mpc_model.getJointVelocities(x_plan, u_plan);

    tau.setZero();
    double q_err_sq = 0.0;
    for (size_t m = 0; m < joint_names.size(); ++m) {
      const int a = mpc_to_act[m];
      if (a < 0) continue;
      double q = 0, qd = 0;
      try {
        const auto& j = dynamic_cast<const mbd::RevoluteJoint<double>&>(
            plant.GetJointByName(joint_names[m]));
        q = j.get_angle(plant_ctx);
        qd = j.get_angular_rate(plant_ctx);
      } catch (...) {}

      const double qp = q_plan[static_cast<Eigen::Index>(m)];
      const double qdp = std::clamp(
          qd_plan[static_cast<Eigen::Index>(m)], -0.5, 0.5);  // same clamp as bridge

      const std::string& jn = joint_names[m];
      double kp = 90, kd = 5;
      if (jn.find("hip_pitch") != std::string::npos) { kp = 200; kd = 30; }
      else if (jn.find("knee") != std::string::npos) { kp = 80; kd = 8; }
      else if (jn.find("ankle") != std::string::npos) { kp = 60; kd = 5; }
      else if (jn.find("hip") != std::string::npos)   { kp = 140; kd = 10; }

      double ti = kp * (qp - q) + kd * (qdp - qd) - kPassiveDamping * qd;
      ti = std::clamp(ti, -kMaxTorque, kMaxTorque);
      tau[a] = ti;
      q_err_sq += (qp - q) * (qp - q);
    }

    plant.get_actuation_input_port(model_instance).FixValue(&plant_ctx, tau);

    t += kSimDt;
    sim.AdvanceTo(t);

    const auto X = plant.EvalBodyPoseInWorld(plant_ctx, pelvis);
    const auto R = X.rotation().matrix();
    const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
    const double pz = X.translation().z();
    fell = (pz < 0.35) || (std::abs(pitch) > 0.9);

    if ((step++ % log_every) == 0 || fell) {
      std::printf("%8.3f %10.4f %10.2f %10.4f %10.1f %10s\n",
                  t, pz, pitch * 57.2958,
                  std::sqrt(q_err_sq / joint_names.size()),
                  tau.cwiseAbs().maxCoeff(),
                  fell ? "FELL" : "ok");
    }
  }

  std::printf("\n================ PLAYBACK VERDICT ================\n");
  if (fell) {
    std::printf("Robot fell at t=%.3f s under the MPC's OWN planned torques.\n"
                "-> Policy is wrong for Drake dynamics. Look at:\n"
                "   - centroidal-vs-rigid-body model gap (inertias, contact)\n"
                "   - Stage 0 force normalization (Fz printed above)\n"
                "   - foot geometry (run static_equilibrium_check first)\n", t);
    return 2;
  }
  std::printf("Robot tracked the plan for %.2f s without falling.\n"
              "-> Policy is sound open-loop. If closed loop still falls, the\n"
              "   problem is timing/staleness: decouple physics thread or\n"
              "   lower mpcDesiredFrequency until overruns stop.\n", t);
  return 0;
}
