/******************************************************************************
kinematic_replay_test.cpp  —  STAGE 1: KINEMATIC REPLAY (no dynamics)
(rev 2: solve via SqpMpc — the API this fork actually exposes)

Answers ONE question: "Do the MPC's coordinate conventions match Drake's?"
See PASS criteria printed at the end.

Usage:
  kinematic_replay_test g1 task.info reference.info g1_29dof.urdf gait.info \
                        g1_29dof_drake.xml [loops=3] [speed=0.5]
******************************************************************************/

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <drake/geometry/meshcat.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/scene_graph.h>
#include <drake/math/rigid_transform.h>
#include <drake/math/rotation_matrix.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/revolute_joint.h>
#include <drake/systems/framework/diagram_builder.h>

#include <ocs2_core/misc/LinearInterpolation.h>
#include <ocs2_sqp/SqpMpc.h>

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h>

namespace mbd = drake::multibody;
namespace geo = drake::geometry;
using namespace ocs2;
using namespace ocs2::humanoid;

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr,
        "Usage: kinematic_replay_test robot task ref urdf gait drake_model "
        "[loops=3] [speed=0.5]\n");
    return 1;
  }
  const std::string task_file = argv[2];
  const std::string reference_file = argv[3];
  const std::string urdf_file = argv[4];
  const std::string drake_model = argv[6];
  const int loops = (argc > 7) ? std::stoi(argv[7]) : 3;
  const double speed = (argc > 8) ? std::stod(argv[8]) : 0.5;  // 0.5 = half speed

  /* ----------------------- 1. Solve MPC once --------------------------- */
  CentroidalMpcInterface interface(task_file, urdf_file, reference_file);
  const auto& mpc_model = interface.getMpcRobotModel();
  const auto& joint_names = interface.modelSettings().mpcModelJointNames;

  const vector_t x0 = interface.getInitialState();
  const scalar_t horizon = interface.mpcSettings().timeHorizon_;

  // Stand-in-place target: zero commanded velocity from x0.
  CentroidalMpcTargetTrajectoriesCalculator target_calc(
      reference_file, interface.getMpcRobotModel(), interface.getPinocchioInterface(),
      interface.getCentroidalModelInfo(), horizon);
  auto target = target_calc.commandedVelocityToTargetTrajectories(
      vector4_t::Zero(), 0.0, x0);
  interface.getReferenceManagerPtr()->setTargetTrajectories(std::move(target));

  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(),
             interface.getOptimalControlProblem(), interface.getInitializer());
  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());

  mpc.run(0.0, x0);

  PrimalSolution primal;
  mpc.getSolverPtr()->getPrimalSolution(horizon, &primal);
  std::printf("Primal: %zu knots over [%.2f, %.2f] s\n",
              primal.timeTrajectory_.size(),
              primal.timeTrajectory_.front(), primal.timeTrajectory_.back());

  /* -------------- 2. Drake plant: visualization only ------------------- */
  drake::systems::DiagramBuilder<double> builder;
  // Continuous plant, never stepped — purely a kinematic display rig.
  auto [plant, scene_graph] = mbd::AddMultibodyPlantSceneGraph(&builder, 0.0);
  mbd::Parser parser(&plant);
  parser.AddModels(drake_model);
  plant.Finalize();

  auto meshcat = std::make_shared<geo::Meshcat>();
  geo::MeshcatVisualizer<double>::AddToBuilder(&builder, scene_graph, meshcat);
  std::printf("Meshcat: %s\n", meshcat->web_url().c_str());

  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  auto& plant_ctx = diagram->GetMutableSubsystemContext(plant, context.get());

  const auto& pelvis = plant.GetBodyByName("pelvis");

  /* ----------------- 3. Replay frames, collect checks ------------------ */
  // EULER NOTE: MPC state layout is [h(6), base_pos(3), euler_ZYX(3), q(23)].
  // OCS2 centroidal convention: euler_ZYX = [yaw(z), pitch(y), roll(x)],
  //   R = Rz(yaw) * Ry(pitch) * Rx(roll).
  // If the robot appears tilted/rotated at t=0 in Meshcat, THIS is the
  // suspect — try swapping to [roll, pitch, yaw] ordering below.
  constexpr int kBasePosIdx = 6;
  constexpr int kEulerIdx = 9;

  const double dt_frame = 1.0 / 60.0;          // 60 fps display
  const double t_end = primal.timeTrajectory_.back() - 1e-3;

  double ankle_z_min = std::numeric_limits<double>::max();
  double ankle_z_max = std::numeric_limits<double>::lowest();
  double base_z_min = ankle_z_min, base_z_max = ankle_z_max;
  int limit_violations = 0;

  for (int loop = 0; loop < loops; ++loop) {
    std::printf("Replay loop %d/%d ...\n", loop + 1, loops);
    for (double t = 0.0; t < t_end; t += dt_frame * speed) {
      const vector_t x = LinearInterpolation::interpolate(
          t, primal.timeTrajectory_, primal.stateTrajectory_);

      // Base pose.
      const Eigen::Vector3d p = x.segment<3>(kBasePosIdx);
      const Eigen::Vector3d eul = x.segment<3>(kEulerIdx);  // [yaw, pitch, roll]
      const drake::math::RotationMatrixd R =
          drake::math::RotationMatrixd::MakeZRotation(eul[0]) *
          drake::math::RotationMatrixd::MakeYRotation(eul[1]) *
          drake::math::RotationMatrixd::MakeXRotation(eul[2]);
      plant.SetFreeBodyPose(&plant_ctx, pelvis,
                            drake::math::RigidTransformd(R, p));

      // Joints.
      const vector_t q = mpc_model.getJointAngles(x);
      for (size_t m = 0; m < joint_names.size(); ++m) {
        try {
          const auto& j = dynamic_cast<const mbd::RevoluteJoint<double>&>(
              plant.GetJointByName(joint_names[m]));
          const double qm = q[static_cast<Eigen::Index>(m)];
          j.set_angle(&plant_ctx, qm);
          if (qm < j.position_lower_limit() || qm > j.position_upper_limit())
            ++limit_violations;
        } catch (...) {}
      }

      diagram->ForcedPublish(*context);

      // Convention metrics (collect on first loop only).
      if (loop == 0) {
        base_z_min = std::min(base_z_min, p.z());
        base_z_max = std::max(base_z_max, p.z());
        for (const auto& an : {std::string("left_ankle_roll_link"),
                               std::string("right_ankle_roll_link")}) {
          try {
            const double az = plant.EvalBodyPoseInWorld(
                plant_ctx, plant.GetBodyByName(an)).translation().z();
            ankle_z_min = std::min(ankle_z_min, az);
            ankle_z_max = std::max(ankle_z_max, az);
          } catch (...) {}
        }
      }

      std::this_thread::sleep_for(
          std::chrono::duration<double>(dt_frame));
    }
  }

  /* --------------------------- 4. Verdict ------------------------------ */
  std::printf("\n================ KINEMATIC REPLAY CHECKS ================\n");
  std::printf("base z   range over plan: [%.4f, %.4f]  (spread %.4f m)\n",
              base_z_min, base_z_max, base_z_max - base_z_min);
  std::printf("ankle z  range over plan: [%.4f, %.4f]  (spread %.4f m)\n",
              ankle_z_min, ankle_z_max, ankle_z_max - ankle_z_min);
  std::printf("joint limit violations  : %d\n", limit_violations);
  std::printf(
      "\nPASS criteria:\n"
      "  - In Meshcat, robot stands upright, faces +x, knees bend forward.\n"
      "  - Ankle z spread small (< ~0.02 m for standing; larger only if the\n"
      "    plan actually lifts a foot).\n"
      "  - Zero joint limit violations.\n"
      "If the robot is visibly tilted at t=0 -> Euler convention mismatch\n"
      "(see EULER NOTE in source). If a limb folds wrong -> joint sign or\n"
      "ordering mismatch between MPC joint list and Drake model.\n");
  return 0;
}
