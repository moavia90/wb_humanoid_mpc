/******************************************************************************
threaded_closed_loop_test.cpp  —  STAGE 5: CLOSED LOOP WITH HONEST TIMING

Full loop (Drake state -> MPC/MRT -> torques -> Drake), restructured with
the MuJoCo interface's threading pattern so a slow MPC solve can never
produce stale physics:

  PHYSICS THREAD (this file's core difference vs drake_mrt_sim_node):
    - steps Drake at fixed sim_dt with sleep_until pacing (MuJoCo's
      simulationLoop pattern), completely decoupled from the controller
    - each step: snapshot latest JointAction under mutex, compute
      tau = getTotalFeedbackTorque(q, qd) - d*qd, clamp, FixValue, AdvanceTo
    - updates the shared RobotState after every step
    - fall reset: restore init state, zero torque, 1.0 s cooldown
      (mirrors MujocoSimInterface::simulationStep)

  CONTROL THREAD (main):
    - loops at mrt rate, copies the shared RobotState, calls
      controller.computeJointControlAction(...), writes the resulting
      JointAction back under mutex
    - if the MPC overruns, physics simply keeps integrating with the most
      recent action — exactly what MuJoCo does — instead of stalling

Run ONLY after Stages 0-4 pass. Then a fall here means closed-loop
interaction (model drift under feedback), not geometry, not policy, not
timing artifacts.

Same 6 positional args as drake_mrt_sim_node.
******************************************************************************/

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>

#include <Eigen/Dense>

#include <drake/geometry/meshcat.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/proximity_properties.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/shape_specification.h>
#include <drake/math/rigid_transform.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/revolute_joint.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram_builder.h>

#include <ocs2_sqp/SqpMpc.h>

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h>
#include <humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h>
#include <humanoid_common_mpc_ros2/ros_comm/Ros2ProceduralMpcMotionManager.h>
#include <humanoid_common_mpc_ros2/visualization/HumanoidVisualizer.h>

#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>

namespace mbd = drake::multibody;
namespace geo = drake::geometry;
using namespace ocs2;
using namespace ocs2::humanoid;

namespace {

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

}  // namespace

/* ========================================================================== */
/* Threaded Drake physics — MuJoCo simulationLoop pattern                     */
/* ========================================================================== */

class ThreadedDrakeSim {
 public:
  ThreadedDrakeSim(const robot::model::RobotDescription& desc,
                   const robot::model::RobotState& init_state,
                   const std::string& model_path, double sim_dt)
      : desc_(desc),
        shared_state_(desc, 2),
        shared_action_(desc),
        init_state_(init_state),
        model_path_(model_path),
        sim_dt_(sim_dt) {
    // NOTE: Drake construction is deferred to the physics thread (see
    // loop()). Drake's Meshcat aborts if any of its methods -- including
    // the publishes inside Simulator::AdvanceTo -- run on a different
    // thread than the one that constructed it. Building everything on the
    // physics thread keeps construction and publishing on one thread.
  }

  ~ThreadedDrakeSim() {
    terminate_.store(true);
    if (thread_.joinable()) thread_.join();
  }

  void start() {
    thread_ = std::thread(&ThreadedDrakeSim::loop, this);
    std::unique_lock<std::mutex> lk(build_mtx_);
    build_cv_.wait(lk, [this] { return built_.load(); });
  }

  // Controller-side accessors (thread-safe copies).
  robot::model::RobotState copyState() {
    std::lock_guard<std::mutex> g(mtx_);
    return shared_state_;
  }
  void setAction(const robot::model::RobotJointAction& a) {
    std::lock_guard<std::mutex> g(mtx_);
    shared_action_ = a;
    have_action_ = true;
  }
  void markControllerReady() { controller_ready_.store(true); }

  double simTime() const { return sim_time_.load(); }
  int resetCount() const { return reset_count_.load(); }
  double physicsRtf() const { return rtf_.load(); }

 private:
  void build(const std::string& model_path) {
    drake::systems::DiagramBuilder<double> builder;
    auto pair = mbd::AddMultibodyPlantSceneGraph(&builder, sim_dt_);
    plant_ = &pair.plant;
    plant_->set_contact_model(mbd::ContactModel::kPointContactOnly);
    plant_->set_discrete_contact_approximation(
        mbd::DiscreteContactApproximation::kSap);

    mbd::Parser parser(plant_);
    parser.package_map().PopulateFromEnvironment("AMENT_PREFIX_PATH");
    mi_ = parser.AddModels(model_path).at(0);

    geo::ProximityProperties gp;
    geo::AddContactMaterial(2.0, 5e4,
                            mbd::CoulombFriction<double>(1.2, 1.0), &gp);
    plant_->RegisterCollisionGeometry(plant_->world_body(),
        drake::math::RigidTransformd(Eigen::Vector3d(0, 0, -0.1)),
        geo::Box(20, 20, 0.2), "ground_collision", gp);
    plant_->RegisterVisualGeometry(plant_->world_body(),
        drake::math::RigidTransformd(Eigen::Vector3d(0, 0, -0.1)),
        geo::Box(20, 20, 0.2), "ground_visual",
        Eigen::Vector4d(0.5, 0.5, 0.5, 1.0));

    plant_->Finalize();

    meshcat_ = std::make_shared<geo::Meshcat>();
    geo::MeshcatVisualizerParams vp;
    vp.publish_period = 1.0 / 30.0;
    geo::MeshcatVisualizer<double>::AddToBuilder(&builder, pair.scene_graph,
                                                 meshcat_, vp);
    std::printf("Meshcat: %s\n", meshcat_->web_url().c_str());

    diagram_ = builder.Build();
    sim_ = std::make_unique<drake::systems::Simulator<double>>(*diagram_);
    sim_->set_target_realtime_rate(0.0);

    n_act_ = plant_->num_actuators();
    tau_ = Eigen::VectorXd::Zero(n_act_);
    for (mbd::JointActuatorIndex i(0); i < n_act_; ++i)
      act_names_.push_back(plant_->get_joint_actuator(i).joint().name());
  }

  drake::systems::Context<double>& plantCtx() {
    return diagram_->GetMutableSubsystemContext(*plant_,
                                                &sim_->get_mutable_context());
  }

  void setState(const robot::model::RobotState& s) {
    auto& ctx = plantCtx();
    const auto& pelvis = plant_->GetBodyByName("pelvis");
    const auto p = s.getRootPositionInWorldFrame();
    plant_->SetFreeBodyPose(&ctx, pelvis,
        drake::math::RigidTransformd(
            drake::math::RotationMatrixd(
                s.getRootRotationLocalToWorldFrame().toRotationMatrix()),
            Eigen::Vector3d(p[0], p[1], p[2])));
    mbd::SpatialVelocity<double> zero; zero.SetZero();
    plant_->SetFreeBodySpatialVelocity(&ctx, pelvis, zero);

    for (const auto& jn : act_names_) {
      double q0 = nominalQ(jn);
      if (desc_.containsJoint(jn))
        q0 = s.getJointPosition(desc_.getJointIndex(jn));
      try {
        const auto& j = dynamic_cast<const mbd::RevoluteJoint<double>&>(
            plant_->GetJointByName(jn));
        j.set_angle(&ctx, q0);
        j.set_angular_rate(&ctx, 0.0);
      } catch (...) {}
    }
    plant_->get_actuation_input_port(mi_).FixValue(&ctx,
        Eigen::VectorXd::Zero(n_act_));
  }

  void snapshotInit() {
    // The fall-reset restore target, MuJoCo qpos_init_/qvel_init_ style.
    // Here we just re-apply init_state_ on reset, so nothing extra to copy.
  }

  void updateSharedState() {
    const auto& ctx = diagram_->GetSubsystemContext(*plant_, sim_->get_context());
    const auto& pelvis = plant_->GetBodyByName("pelvis");
    const auto X = plant_->EvalBodyPoseInWorld(ctx, pelvis);
    const auto V = plant_->EvalBodySpatialVelocityInWorld(ctx, pelvis);

    robot::quaternion_t q_lw(X.rotation().matrix());

    std::lock_guard<std::mutex> g(mtx_);
    shared_state_.setRootRotationLocalToWorldFrame(q_lw);
    shared_state_.setRootPositionInWorldFrame(X.translation());
    shared_state_.setRootLinearVelocityInLocalFrame(
        q_lw.inverse() * V.translational());
    shared_state_.setRootAngularVelocityInLocalFrame(
        q_lw.inverse() * V.rotational());

    for (const auto& jn : desc_.getJointNames()) {
      const auto idx = desc_.getJointIndex(jn);
      try {
        const auto& j = dynamic_cast<const mbd::RevoluteJoint<double>&>(
            plant_->GetJointByName(jn));
        shared_state_.setJointPosition(idx, j.get_angle(ctx));
        shared_state_.setJointVelocity(idx, j.get_angular_rate(ctx));
      } catch (...) {
        shared_state_.setJointPosition(idx, 0.0);
        shared_state_.setJointVelocity(idx, 0.0);
      }
    }
    // MuJoCo-equivalent forced contact flags.
    shared_state_.setContactFlag(0, true);
    shared_state_.setContactFlag(1, true);
    shared_state_.setTime(sim_time_.load());
  }

  bool fallen() {
    const auto& ctx = diagram_->GetSubsystemContext(*plant_, sim_->get_context());
    const auto X = plant_->EvalBodyPoseInWorld(ctx,
                                               plant_->GetBodyByName("pelvis"));
    const auto R = X.rotation().matrix();
    const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
    const double roll = std::atan2(R(2, 1), R(2, 2));
    return X.translation().z() < 0.25 ||
           std::abs(pitch) > 0.9 || std::abs(roll) > 0.9;
  }

  /* ---------------- THE MUJOCO PATTERN: dedicated physics loop ---------- */
  void loop() {
    // ---- Drake construction ON the physics thread (Meshcat same-thread
    // requirement; see constructor note). ----
    build(model_path_);
    setState(init_state_);
    sim_->Initialize();
    snapshotInit();
    updateSharedState();
    {
      std::lock_guard<std::mutex> lk(build_mtx_);
      built_.store(true);
    }
    build_cv_.notify_all();

    constexpr double kPassiveDamping = 10.0;
    constexpr double kPinDuration = 2.0;       // warm-start pin
    auto next_wakeup = std::chrono::steady_clock::now();
    auto last_real = std::chrono::steady_clock::now();
    const auto step_us = std::chrono::microseconds(
        static_cast<long>(sim_dt_ * 1e6));

    robot::model::RobotJointAction action(desc_);
    double pin_release_t = -1.0;

    while (!terminate_.load()) {
      auto& ctx = plantCtx();

      const bool pinned =
          !controller_ready_.load() ||
          (pin_release_t < 0.0) ||
          (sim_time_.load() < pin_release_t);

      if (!controller_ready_.load()) {
        // keep holding spawn state until MPC has a policy
        setState(init_state_);
      } else if (pin_release_t < 0.0) {
        pin_release_t = sim_time_.load() + kPinDuration;
        std::printf("[physics] controller ready; pin releases at t=%.2f\n",
                    pin_release_t);
      }

      if (pinned) {
        if (controller_ready_.load()) setState(init_state_);
        tau_.setZero();
      } else {
        // Snapshot latest action under mutex (MuJoCo copy_value pattern),
        // then compute torques from CURRENT Drake state. If the MPC is
        // slow, this is simply the most recent action — physics never waits.
        {
          std::lock_guard<std::mutex> g(mtx_);
          if (have_action_) action = shared_action_;
        }
        for (int a = 0; a < n_act_; ++a) {
          const std::string& jn = act_names_[a];
          double t_ctrl = 0, q = 0, qd = 0;
          try {
            const auto& j = dynamic_cast<const mbd::RevoluteJoint<double>&>(
                plant_->GetJointByName(jn));
            q = j.get_angle(ctx);
            qd = j.get_angular_rate(ctx);
          } catch (...) {}

          if (desc_.containsJoint(jn)) {
            const auto idx = desc_.getJointIndex(jn);
            const auto& opt = action.at(idx);
            if (opt.has_value())
              t_ctrl = opt.value().getTotalFeedbackTorque(q, qd);
          }
          if (!std::isfinite(t_ctrl)) t_ctrl = 0.0;

          const double lim = torqueLimit(jn);
          tau_[a] = std::clamp(t_ctrl - kPassiveDamping * qd, -lim, lim);
        }
      }

      plant_->get_actuation_input_port(mi_).FixValue(&ctx, tau_);

      const double t_new = sim_time_.load() + sim_dt_;
      try {
        sim_->AdvanceTo(t_new);
        sim_time_.store(t_new);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[physics] AdvanceTo failed: %s -> reset\n",
                     e.what());
        doReset();
        continue;
      }

      updateSharedState();

      // rtf metric
      const auto now = std::chrono::steady_clock::now();
      const double real_dt =
          std::chrono::duration<double>(now - last_real).count();
      last_real = now;
      rtf_.store(sim_dt_ / std::max(1e-9, real_dt));

      // MuJoCo-style fall reset with 1 s controller-adjust dwell.
      if (!pinned && fallen()) {
        std::printf("[physics] fall at t=%.2f -> reset (count=%d)\n",
                    sim_time_.load(), reset_count_.load() + 1);
        doReset();
        pin_release_t = sim_time_.load() + kPinDuration;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        next_wakeup = std::chrono::steady_clock::now();
        continue;
      }

      next_wakeup += step_us;
      std::this_thread::sleep_until(next_wakeup);
    }
  }

  void doReset() {
    setState(init_state_);
    tau_.setZero();
    plant_->get_actuation_input_port(mi_).FixValue(&plantCtx(), tau_);
    try { sim_->Initialize(); } catch (...) {}
    updateSharedState();
    reset_count_.fetch_add(1);
  }

  /* members */
  const robot::model::RobotDescription& desc_;
  robot::model::RobotState shared_state_;
  robot::model::RobotJointAction shared_action_;
  robot::model::RobotState init_state_;
  bool have_action_{false};

  double sim_dt_;
  std::atomic<double> sim_time_{0.0};
  std::atomic<double> rtf_{0.0};
  std::atomic<bool> controller_ready_{false};
  std::atomic<bool> terminate_{false};
  std::atomic<int> reset_count_{0};

  std::mutex mtx_;
  std::thread thread_;

  std::string model_path_;
  std::mutex build_mtx_;
  std::condition_variable build_cv_;
  std::atomic<bool> built_{false};

  mbd::MultibodyPlant<double>* plant_{nullptr};
  mbd::ModelInstanceIndex mi_;
  std::shared_ptr<geo::Meshcat> meshcat_;
  std::unique_ptr<drake::systems::Diagram<double>> diagram_;
  std::unique_ptr<drake::systems::Simulator<double>> sim_;
  std::vector<std::string> act_names_;
  Eigen::VectorXd tau_;
  int n_act_{0};
};

/* ========================================================================== */
/* main: control thread at MRT rate, physics fully decoupled                  */
/* ========================================================================== */

int main(int argc, char** argv) {
  const auto args = rclcpp::remove_ros_arguments(argc, argv);
  if (args.size() < 7) {
    std::fprintf(stderr,
        "Usage: threaded_closed_loop_test robot task ref urdf gait drake_model\n");
    return 1;
  }
  const std::string robot_name = args[1], task_file = args[2],
                    reference_file = args[3], urdf_file = args[4],
                    gait_file = args[5], drake_model = args[6];

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(robot_name + "_threaded_cl_test");

  constexpr double kSimDt = 0.0005;

  /* ---- MPC stack: identical construction to drake_mrt_sim_node -------- */
  CentroidalMpcInterface interface(task_file, urdf_file, reference_file);
  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(),
             interface.getOptimalControlProblem(), interface.getInitializer());

  auto qos = rclcpp::QoS(1).best_effort();
  auto viz = std::make_shared<HumanoidVisualizer>(
      task_file, interface.getPinocchioInterface(),
      interface.getMpcRobotModel(), node);

  CentroidalMpcTargetTrajectoriesCalculator target_calc(
      reference_file, interface.getMpcRobotModel(),
      interface.getPinocchioInterface(), interface.getCentroidalModelInfo(),
      interface.mpcSettings().timeHorizon_);
  ProceduralMpcMotionManager::VelocityTargetToTargetTrajectories target_func =
      [&target_calc](const vector4_t& v, scalar_t t0, scalar_t tf,
                     const vector_t& x) mutable {
        (void)tf;
        return target_calc.commandedVelocityToTargetTrajectories(v, t0, x);
      };
  auto motion_manager = std::make_shared<Ros2ProceduralMpcMotionManager>(
      gait_file, reference_file, interface.getSwitchedModelReferenceManagerPtr(),
      interface.getMpcRobotModel(), target_func);
  motion_manager->subscribe(node, qos);
  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());
  mpc.getSolverPtr()->addSynchronizedModule(motion_manager);

  /* ---- Initial state from MPC (same as main node) ---------------------- */
  robot::model::RobotDescription desc(urdf_file);
  robot::model::RobotState init_state(desc, 2);
  init_state.setConfigurationToZero();

  const vector_t x0 = interface.getInitialState();
  const auto& mpc_model = interface.getMpcRobotModel();
  init_state.setRootPositionInWorldFrame(mpc_model.getBasePosition(x0));
  const vector_t q0 = mpc_model.getJointAngles(x0);
  const auto idxs =
      desc.getJointIndices(interface.modelSettings().mpcModelJointNames);
  for (size_t i = 0; i < idxs.size(); ++i)
    init_state.setJointPosition(idxs[i], q0[static_cast<Eigen::Index>(i)]);

  /* ---- Physics on its own thread --------------------------------------- */
  ThreadedDrakeSim sim(desc, init_state, drake_model, kSimDt);
  sim.start();

  CentroidalMpcMrtJointController controller(
      desc, interface.modelSettings(), interface.getMpcRobotModel(), mpc,
      interface.getPinocchioInterface(),
      interface.mpcSettings().mpcDesiredFrequency_, viz);

  {
    auto s = sim.copyState();
    controller.startMpcThread(s);
  }
  while (rclcpp::ok() && !controller.ready()) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  sim.markControllerReady();
  RCLCPP_INFO(node->get_logger(), "MRT ready; control loop starting.");

  /* ---- Control loop at MRT rate. Physics NEVER waits for this. --------- */
  const double mrt_hz = interface.mpcSettings().mrtDesiredFrequency_ > 0
                            ? interface.mpcSettings().mrtDesiredFrequency_
                            : 100.0;
  const auto period = std::chrono::microseconds(
      static_cast<long>(1e6 / mrt_hz));
  robot::model::RobotJointAction action(desc);

  int n = 0;
  auto next = std::chrono::steady_clock::now();
  while (rclcpp::ok()) {
    const auto t0 = std::chrono::steady_clock::now();

    auto state = sim.copyState();
    controller.computeJointControlAction(state.getTime(), state, action);
    sim.setAction(action);

    rclcpp::spin_some(node);

    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    const double budget = 1000.0 / mrt_hz;
    if ((n++ % static_cast<int>(mrt_hz)) == 0) {
      RCLCPP_INFO(node->get_logger(),
          "[timing] sim_t=%.2f ctrl=%.2fms budget=%.2fms %s | "
          "physics_rtf=%.2f resets=%d",
          sim.simTime(), ms, budget,
          ms > budget ? "OVERRUN (physics unaffected)" : "ok",
          sim.physicsRtf(), sim.resetCount());
    }

    next += period;
    std::this_thread::sleep_until(next);
    if (std::chrono::steady_clock::now() > next + period) {
      next = std::chrono::steady_clock::now();  // resync after big stall
    }
  }

  rclcpp::shutdown();
  return 0;
}
