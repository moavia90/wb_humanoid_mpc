/******************************************************************************
static_equilibrium_check.cpp  —  STAGE 2 GATE   (rev 3)

rev 3 fixes a fatal harness bug in rev 2: foot geometries were matched by
GEOMETRY name, but Drake's MJCF parser does not name them the way we
assumed, so ALL 703 of 703 collision pairs were ignored — including
foot-ground — making "infeasible" a foregone conclusion.

Now:
  - geometries are matched by the BODY they are attached to
    (body name contains "ankle_roll" -> foot; world body / "ground" -> ground)
  - the full proximity-geometry inventory (name + body) is printed
  - at the seed pose, the lowest point of every foot collision geometry is
    printed -> directly measures spawn penetration (the 2 cm pop seen in
    Stage 4) or floating feet
  - total robot mass is printed so the expected Fz is exact, not assumed

Usage:
  static_equilibrium_check <drake_model.xml> [pelvis_z=0.78]
Exit: 0 = feasible (gate passed), 2 = infeasible.
******************************************************************************/

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <drake/geometry/proximity_properties.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/shape_specification.h>
#include <drake/math/rigid_transform.h>
#include <drake/multibody/optimization/static_equilibrium_problem.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/revolute_joint.h>
#include <drake/solvers/solve.h>
#include <drake/systems/framework/diagram_builder.h>

namespace mbd = drake::multibody;
namespace geo = drake::geometry;

// Same nominal stance the sim node uses.
double nominalQ(const std::string& j) {
  if (j.find("hip_pitch") != std::string::npos) return -0.20;
  if (j.find("knee") != std::string::npos) return 0.40;
  if (j.find("ankle_pitch") != std::string::npos) return -0.20;
  return 0.0;
}

template <typename T>
void seedPose(const mbd::MultibodyPlant<T>& plant,
              drake::systems::Context<T>* ctx, double pelvis_z) {
  const auto& pelvis = plant.GetBodyByName("pelvis");
  plant.SetFreeBodyPose(ctx, pelvis,
      drake::math::RigidTransform<T>(
          drake::math::RotationMatrix<T>::Identity(),
          drake::Vector3<T>(T(0.0), T(0.0), T(pelvis_z))));
  for (mbd::JointIndex ji(0); ji < plant.num_joints(); ++ji) {
    const auto& joint = plant.get_joint(ji);
    if (joint.num_positions() != 1) continue;
    const auto* rj = dynamic_cast<const mbd::RevoluteJoint<T>*>(&joint);
    if (rj) rj->set_angle(ctx, T(nominalQ(joint.name())));
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: static_equilibrium_check <drake_model.xml> [pelvis_z=0.78]\n";
    return 1;
  }
  const std::string model_path = argv[1];
  const double pelvis_z = (argc > 2) ? std::stod(argv[2]) : 0.78;

  /* ---- Build double plant (parsing), keep for diagnostics --------------- */
  drake::systems::DiagramBuilder<double> builder;
  auto [plant_d, scene_graph] =
      mbd::AddMultibodyPlantSceneGraph(&builder, 0.001);

  mbd::Parser parser(&plant_d);
  parser.package_map().PopulateFromEnvironment("AMENT_PREFIX_PATH");
  parser.AddModels(model_path);

  geo::ProximityProperties gp;
  geo::AddContactMaterial(2.0, 5e4,
      mbd::CoulombFriction<double>(1.2, 1.0), &gp);
  plant_d.RegisterCollisionGeometry(
      plant_d.world_body(),
      drake::math::RigidTransformd(Eigen::Vector3d(0, 0, -0.1)),
      geo::Box(20, 20, 0.2), "ground_collision", gp);

  plant_d.Finalize();
  auto diagram_d = builder.Build();

  /* ---- Classify geometries BY BODY, print inventory --------------------- */
  const auto& inspector = scene_graph.model_inspector();
  const auto all_geoms = inspector.GetAllGeometryIds(geo::Role::kProximity);

  auto body_of = [&](geo::GeometryId id) -> const mbd::RigidBody<double>* {
    return plant_d.GetBodyFromFrameId(inspector.GetFrameId(id));
  };
  auto is_foot = [&](geo::GeometryId id) {
    const auto* b = body_of(id);
    return b && b->name().find("ankle_roll") != std::string::npos;
  };
  auto is_ground = [&](geo::GeometryId id) {
    const auto* b = body_of(id);
    if (b && b->index() == plant_d.world_body().index()) return true;
    return inspector.GetName(id).find("ground") != std::string::npos;
  };

  std::printf("---- Proximity geometry inventory (%zu) ----\n", all_geoms.size());
  for (const auto& gid : all_geoms) {
    const auto* b = body_of(gid);
    const char* tag = is_foot(gid) ? " [FOOT]" : (is_ground(gid) ? " [GROUND]" : "");
    std::printf("  body=%-28s geom=%s%s\n",
                b ? b->name().c_str() : "<?>",
                inspector.GetName(gid).c_str(), tag);
  }

  /* ---- Spawn-penetration diagnostic at the seed pose --------------------- */
  {
    auto ctx = diagram_d->CreateDefaultContext();
    auto& pctx = diagram_d->GetMutableSubsystemContext(plant_d, ctx.get());
    seedPose(plant_d, &pctx, pelvis_z);
    std::printf("\nTotal robot mass: %.2f kg -> expected total Fz = %.1f N\n",
                plant_d.CalcTotalMass(pctx),
                plant_d.CalcTotalMass(pctx) * 9.81);
    std::printf("---- Foot geometry lowest points at seed pose (ground at z=0) ----\n");
    for (const auto& gid : all_geoms) {
      if (!is_foot(gid)) continue;
      const auto* b = body_of(gid);
      const auto X_WB = plant_d.EvalBodyPoseInWorld(pctx, *b);
      const auto X_WG = X_WB * inspector.GetPoseInFrame(gid);
      const geo::Shape& shape = inspector.GetShape(gid);
      double low = X_WG.translation().z();
      std::string kind = "other";
      if (const auto* s = dynamic_cast<const geo::Sphere*>(&shape)) {
        low -= s->radius();
        kind = "sphere r=" + std::to_string(s->radius());
      } else if (const auto* bx = dynamic_cast<const geo::Box*>(&shape)) {
        kind = "box";
        const Eigen::Vector3d h(bx->width() / 2, bx->depth() / 2, bx->height() / 2);
        low = 1e9;
        for (int i = 0; i < 8; ++i) {
          const Eigen::Vector3d c((i & 1 ? h.x() : -h.x()),
                                  (i & 2 ? h.y() : -h.y()),
                                  (i & 4 ? h.z() : -h.z()));
          low = std::min(low, (X_WG * c).z());
        }
      } else if (const auto* cp = dynamic_cast<const geo::Capsule*>(&shape)) {
        kind = "capsule r=" + std::to_string(cp->radius());
        const Eigen::Vector3d a = X_WG * Eigen::Vector3d(0, 0,  cp->length() / 2);
        const Eigen::Vector3d bq = X_WG * Eigen::Vector3d(0, 0, -cp->length() / 2);
        low = std::min(a.z(), bq.z()) - cp->radius();
      }
      std::printf("  %-28s %-22s lowest z = %+.4f m  %s\n",
                  b->name().c_str(), kind.c_str(), low,
                  low < -0.002 ? "<-- PENETRATING"
                               : (low > 0.002 ? "<-- FLOATING" : "(touching)"));
    }
  }

  /* ---- Convert to AutoDiffXd and solve ----------------------------------- */
  auto diagram_ad = drake::systems::System<double>::ToAutoDiffXd(*diagram_d);
  auto context_ad = diagram_ad->CreateDefaultContext();
  const auto& plant_ad = dynamic_cast<const mbd::MultibodyPlant<drake::AutoDiffXd>&>(
      diagram_ad->GetSubsystemByName(plant_d.get_name()));
  auto& plant_context_ad =
      diagram_ad->GetMutableSubsystemContext(plant_ad, context_ad.get());
  seedPose(plant_ad, &plant_context_ad, pelvis_z);

  std::set<std::pair<geo::GeometryId, geo::GeometryId>> ignored;
  int kept = 0;
  for (size_t a = 0; a < all_geoms.size(); ++a) {
    for (size_t b = a + 1; b < all_geoms.size(); ++b) {
      const auto ga = all_geoms[a], gb = all_geoms[b];
      const bool keep = (is_foot(ga) && is_ground(gb)) ||
                        (is_ground(ga) && is_foot(gb));
      if (keep) ++kept; else ignored.insert({ga, gb});
    }
  }
  std::printf("\nPairs: %d kept (foot-ground), %zu ignored.\n", kept, ignored.size());
  if (kept == 0) {
    std::printf("HARNESS ERROR: no foot-ground pairs found -- body-name matching\n"
                "failed. Check the inventory above and adjust is_foot().\n");
    return 3;
  }

  mbd::StaticEquilibriumProblem problem(&plant_ad, &plant_context_ad, ignored);

  const auto q0 = plant_ad.GetPositions(plant_context_ad);
  Eigen::VectorXd q0_d(q0.size());
  for (int i = 0; i < q0.size(); ++i) q0_d[i] = q0[i].value();
  problem.get_mutable_prog()->SetInitialGuess(problem.q_vars(), q0_d);

  const auto result = drake::solvers::Solve(problem.prog());

  std::printf("\n================ STATIC EQUILIBRIUM RESULT ================\n");
  std::printf("Feasible: %s   (solver: %s)\n",
              result.is_success() ? "YES" : "NO",
              result.get_solver_id().name().c_str());

  if (result.is_success()) {
    const Eigen::VectorXd q_sol = result.GetSolution(problem.q_vars());
    std::printf("Solved pelvis position: [%.4f %.4f %.4f]  (seed z was %.4f)\n",
                q_sol[4], q_sol[5], q_sol[6], pelvis_z);

    const auto wrenches = problem.GetContactWrenchSolution(result);
    std::printf("\nContact wrenches (%zu):\n", wrenches.size());
    double total_fz = 0.0;
    for (const auto& w : wrenches) {
      const Eigen::Vector3d F = w.F_Cb_W.translational();
      std::printf("  bodyA=%d bodyB=%d  F=[%7.2f %7.2f %7.2f] N\n",
                  static_cast<int>(w.bodyA_index), static_cast<int>(w.bodyB_index),
                  F[0], F[1], F[2]);
      total_fz += F[2];
    }
    std::printf("Total Fz = %.2f N (compare to expected m*g above)\n", total_fz);
    std::printf("\nGATE PASSED -> physics can support standing.\n");
  } else {
    std::printf("\nGATE FAILED -> nominal pose not statically feasible with this\n"
                "foot collision geometry. Fix before any dynamic test:\n"
                "  - explicit sole geometry (4 corner spheres or sole box)\n"
                "  - COM inside the support polygon; feet touching, not\n"
                "    penetrating/floating (see lowest-point table above)\n");
  }
  return result.is_success() ? 0 : 2;
}
