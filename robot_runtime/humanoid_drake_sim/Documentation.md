# Drake Humanoid Simulation Backend, 2-Node ROS 2 Architecture & Diagnostics Reference

This document provides the definitive architectural reference, physical modeling principles, synchronization protocols, and diagnostic executable catalog for the Unitree G1 humanoid simulation in `humanoid_drake_sim`. It details why the monolithic single-process MuJoCo replica failed in Drake, establishes the production **2-Node ROS 2 Topic-Based OCS2 Policy Architecture** with dedicated CPU core partitioning via `taskset`, details the startup/reset filter warm-up protocol that prevents robot squatting, and documents all 10 diagnostic executables in the package.

---

## 1. Forensic Analysis: MuJoCo Monolithic Replication vs. Drake Realities

### 1.1 The MuJoCo Baseline Mechanics (`CentroidalMpcRobotSim.cpp`)
In the baseline MuJoCo implementation (`src/wb_humanoid_mpc/humanoid_nmpc/humanoid_centroidal_mpc_ros2/src/CentroidalMpcRobotSim.cpp` and `mujoco_sim_interface`), all control and physics routines execute within a single monolithic operating system process:

```text
================================= MuJoCo Baseline =================================
Process: humanoid_centroidal_mpc_sim (Single Monolithic Binary)
  ├── Main Thread (500 Hz):
  │     ├── RobotHWInterfaceBase::updateInterfaceStateFromRobot()
  │     ├── CentroidalMpcMrtJointController::computeJointControlAction()
  │     │     ├── updateMpcObservation() -> Pinocchio centroidal momentum
  │     │     ├── mcpMrtInterface_.evaluatePolicy()
  │     │     └── computeJointTorques<scalar_t>() [Pinocchio inverse dynamics]
  │     ├── RobotHWInterfaceBase::applyJointAction()
  │     └── Ros2ProceduralMpcMotionManager::spin_some()
  │
  ├── MPC Solver Thread (50-100 Hz):
  │     └── SqpMpc::advance() [solves centroidal QP subproblems]
  │
  ├── Dedicated Physics Thread (2000 Hz, dt = 0.0005 s):
  │     └── MujocoSimInterface::simulationLoop() -> mj_step(mjModel, mjData)
  │
  └── Passive Render Thread (60 Hz):
        └── MujocoRenderer::launchRenderThread() -> GLFW / OpenGL viewer
===================================================================================
```

#### Why Monolithic Succeeded in MuJoCo:
1. **Lightweight Physics Compute**: MuJoCo's core stepping function `mj_step` is written in hand-optimized, vectorized C. For the 29-DOF G1 robot with flat ground contact, a single time-step takes only **50–80 $\mu$s**.
2. **Generous CPU Headroom**: In a 500 $\mu$s interval (2000 Hz), MuJoCo's 50–80 $\mu$s execution consumes less than 16% of a single CPU core. The remaining 420–450 $\mu$s leaves ample margin for OS thread scheduling, even when the OCS2 solver runs concurrently.
3. **Automatic Actuator Clamping**: MuJoCo's physics engine strictly clamps `mjData->ctrl` to `actuatorfrcrange` at the solver level, preventing transient control errors from causing numerical explosions.

---

### 1.2 Why Transplanting the Monolithic Pattern to Drake Caused Catastrophic Instability

When this single-process monolithic pattern was replicated in Drake (`drake_mrt_sim_node`), the simulation suffered from immediate robot collapse, violent high-frequency oscillations, and system unresponsiveness. Forensic profiling identified five primary failure mechanisms:

| Mechanism | MuJoCo Baseline | Drake Monolithic Replica | Impact on Simulation Stability |
| :--- | :--- | :--- | :--- |
| **Physics Step Compute Time** | Vectorized C (`mj_step`): **50–80 $\mu$s** | C++ `MultibodyPlant` + SAP contact solver: **300–450 $\mu$s** | In a 500 $\mu$s interval (2000 Hz), Drake leaves only 50–200 $\mu$s margin. Any thread preemption causes missed deadlines. |
| **CPU Core Contention (12 Threads)** | Low CPU footprint; threads fit without contention. | OCS2 SQP solver invokes OpenMP & Eigen across all available cores during QP factorizations. | OpenMP solver worker threads preempt Drake's 2000 Hz physics pacing thread and 500 Hz control loop. |
| **Pacing Jitter & Phase Lag** | Physics thread wakes predictably at 0.5 ms intervals. | Pacing jitter swings from 0.5 ms to 2.5 ms. Control feedback experiences 2–5 ms phase lag. | At $K_p = 1200$, phase lag introduces energy into the system, causing rapid torque chatter and instability. |
| **Actuator Saturation** | Physics engine clamps input torques to motor torque limits. | Drake's `AddJointActuator` stores `effortLimit` as informational metadata; actuation port does **not** clamp. | Jitter-induced errors caused unclamped torque spikes ($> 500-1200$ N$\cdot$m), blowing up the plant dynamics. |
| **Internal Collision Pairs** | Explicitly disabled via `contype="0" conaffinity="0"`. | URDF meshes loaded into SceneGraph created **703 self-collision pairs**. | Spurious internal contact penalty forces between adjacent links destabilized the robot mechanism. |

#### OpenMP Preemption of 2000 Hz Physics:
On a 12-thread host (Intel Core i5-11400H), when `SqpMpc` solves a sequential quadratic program, OpenMP spawns parallel threads to evaluate dynamics linearizations and factorize QP matrices (HPIPM / qpOASES). In an unpinned monolithic process, these OpenMP threads saturate all 12 CPU cores. When the 2000 Hz physics thread (`std::this_thread::sleep_until(nextWakeup)`) attempts to wake up, the Linux CFS (Completely Fair Scheduler) delays it behind the compute-heavy OpenMP threads. As a result, physics time-steps stall, the robot "hangs" in mid-air, and when the physics thread finally executes, it attempts to catch up with stale joint actions, causing violent impulse shocks.

#### Clarifying the "Legacy Split Pipeline" Trap:
In early development, a two-node experiment (`drake_sim_node` + `policy_to_command_node`) was attempted. That experiment failed not because ROS 2 decoupling is flawed, but because it utilized an ad-hoc custom bridge streaming raw joint torques over `/drake/joint_torques` at high latency (2–5 ms), had QoS mismatches, and required zeroing base velocities. The production solution is **not** to return to a broken monolithic architecture, but to adopt the standard, battle-tested **2-Node ROS 2 Topic-Based OCS2 Policy Architecture**.

---

## 2. Clean 2-Node ROS 2 Topic-Based OCS2 Policy Architecture

The production architecture cleanly decouples the compute-intensive sequential quadratic programming solver from the real-time Drake physics simulation and control loop across two dedicated processes, adhering strictly to Requirements R1, R2, and R3.

```text
========================================================================================================
                                     2-NODE DECOUPLED ARCHITECTURE
========================================================================================================

                                   CORES 0-4 (taskset -c 0-4)
                      ┌──────────────────────────────────────────────────┐
                      │          Node 1: MPC Policy Publisher            │
                      │       (humanoid_centroidal_mpc_sqp_node)         │
                      │                                                  │
                      │  • CentroidalMpcInterface                        │
                      │  • SqpMpc (Centroidal SQP Solver)                │
                      │  • Ros2ProceduralMpcMotionManager                │
                      │  • CentroidalMpcTargetTrajectoriesCalculator     │
                      │  • MPC_ROS_Interface                             │
                      └─────────▲───────────────┬────────────────▲───────┘
                                │               │                │
             /g1/mpc_observation│               │/g1/mpc_policy  │/g1/mpc_reset (Service)
             (MpcObservation)   │               │(MpcFlattened)  │(Reset.srv)
             [QoS: Best Effort] │               │[Best Effort]   │
                                │               │                │
                      ┌─────────┴───────────────▼────────────────┴───────┐
                      │          Node 2: Drake MRT Simulator             │
                      │              (drake_mrt_sim_node)                │
                      │                                                  │
                      │  • MRT_ROS_Interface (OCS2 Policy unpacker)     │
                      │  • 500 Hz MRT Follower Loop                      │
                      │    - Pinocchio inverse dynamics (computeTorques) │
                      │    - Centroidal momentum observation generator   │
                      │  • 2000 Hz DrakeSimInterface (Discrete SAP)      │
                      │    - Dedicated physics thread (dt = 0.5 ms)      │
                      │    - In-memory PD: tau = tau_ff + Kp*e + Kd*ed   │
                      │    - Strict Torque Clamping: |tau| <= max_effort │
                      │  • 60 Hz Decoupled Meshcat Visualizer            │
                      └─────────────────────────┬────────────────────────┘
                                   CORES 5-9 (taskset -c 5-9)
                                                │
                                                │ /humanoid/walking_velocity_command
                                                │ (WalkingVelocityCommand)
                                                │
                      ┌─────────────────────────┴────────────────────────┐
                      │               Cores 10-11 (Uncontended)          │
                      │  • base_velocity_controller_gui (25 Hz Tkinter)  │
                      │  • Meshcat Browser Session (localhost:7000)      │
                      │  • OS / Window Manager / Desktop Shell           │
                      └──────────────────────────────────────────────────┘
========================================================================================================
```

---

### 2.1 Node 1 Specification: OCS2 Policy Publisher (`humanoid_centroidal_mpc_sqp_node`)
- **Package**: `humanoid_centroidal_mpc_ros2` (Strictly unmodified per R3).
- **Executable**: `humanoid_centroidal_mpc_sqp_node`.
- **CPU Pinning**: Cores `0-4` (`taskset -c 0-4`).
- **Core Responsibilities**:
  1. Instantiates `CentroidalMpcInterface` to parse robot kinematics, costs, and constraints.
  2. Executes `SqpMpc` to solve the centroidal optimal control problem over a 1.0 s planning horizon.
  3. `Ros2ProceduralMpcMotionManager` subscribes to `/humanoid/walking_velocity_command`, scales commands, updates gait schedules (`stance` $\rightarrow$ `slow_walk` $\rightarrow$ `walk`), and modifies target reference trajectories before each solve.
  4. `MPC_ROS_Interface` manages ROS 2 communication:
     - Subscribes to `/g1/mpc_observation` (`ocs2_ros2_msgs/msg/MpcObservation`).
     - Publishes `/g1/mpc_policy` (`ocs2_ros2_msgs/msg/MpcFlattenedController`).
     - Hosts `/g1/mpc_reset` (`ocs2_ros2_msgs/srv/Reset`).

---

### 2.2 Node 2 Specification: Drake MRT Policy Follower & Simulator (`drake_mrt_sim_node`)
- **Package**: `humanoid_drake_sim`.
- **Executable**: `drake_mrt_sim_node`.
- **CPU Pinning**: Cores `5-9` (`taskset -c 5-9`).
- **Core Responsibilities**:
  1. Instantiates `ocs2::MRT_ROS_Interface mrt(robotName)` to subscribe to `/g1/mpc_policy`, publish `/g1/mpc_observation`, and invoke `/g1/mpc_reset`.
  2. Controls physics execution via `DrakeSimInterface`:
     - Paces discrete SAP contact solver at 2000 Hz ($dt = 0.0005$ s) on a dedicated physics thread.
     - Evaluates in-memory joint PD feedback: $\tau_i = \tau_{\text{ff},i} + 1200(q_{d,i}-q_i) + 10(\dot{q}_{d,i}-\dot{q}_i)$.
     - Strictly clamps actuator torques to URDF limits: $\tau_i \in [-\tau_{\text{max},i}, \tau_{\text{max},i}]$.
  3. Main Follower Loop at 500 Hz:
     - Copies latest `RobotState` snapshot from `DrakeSimInterface`.
     - Generates `SystemObservation` (calculating centroidal momentum via Pinocchio).
     - Calls `mrt.setCurrentObservation(observation)` to publish `/g1/mpc_observation`.
     - Calls `mrt.updatePolicy()` to unflatten latest linear controller gains and feedforward controls.
     - Evaluates policy at $t_{\text{sim}} + 0.005$ s to obtain target joint states and contact forces.
     - Evaluates Pinocchio inverse dynamics `computeJointTorques` to generate feedforward torques $\tau_{\text{ff}}$.
     - Writes `RobotJointAction` to `DrakeSimInterface`.
  4. Decoupled 60 Hz Meshcat visualizer updating zero-timestep illustration plant.

---

### 2.3 Interface Contracts (Topics & Services)

| Topic / Service | Message Type | QoS | Direction | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `/g1/mpc_observation` | `ocs2_ros2_msgs/msg/MpcObservation` | Best Effort, Depth 1 | Node 2 $\rightarrow$ Node 1 | State feedback: base pose, base twist, joint positions, joint velocities, centroidal momentum. |
| `/g1/mpc_policy` | `ocs2_ros2_msgs/msg/MpcFlattenedController` | Best Effort, Depth 1 | Node 1 $\rightarrow$ Node 2 | Flattened feedback gains $K(t)$ and feedforward controls $u_{\text{ff}}(t)$ over MPC horizon. |
| `/g1/mpc_reset` | `ocs2_ros2_msgs/srv/Reset` | Reliable Service | Node 2 $\rightarrow$ Node 1 | Synchronous reset: re-initializes SQP solver state and resets target trajectories. |
| `/humanoid/walking_velocity_command` | `humanoid_mpc_msgs/msg/WalkingVelocityCommand` | Best Effort, Depth 25 | GUI $\rightarrow$ Node 1 | Teleop inputs: $v_x$, $v_y$, $\omega_z$, and desired pelvis height $h$. |

---

## 3. Dedicated CPU Core Partitioning via `taskset` (Requirement R2)

### 3.1 Hardware Topology & Allocation
The target host is an Intel Core i5-11400H with 6 physical cores and 12 logical CPU threads (`0` through `11`). Under heavy concurrent execution, processes are partitioned as follows:

```text
Cores 0 - 4 (5 threads, 0x1f):   [Node 1: humanoid_centroidal_mpc_sqp_node]
                                 - OCS2 SQP Solver (SqpMpc)
                                 - OpenMP parallel threads for QP subproblems
                                 - Dynamics linearizations & reference management

Cores 5 - 9 (5 threads, 0x3e0):  [Node 2: drake_mrt_sim_node]
                                 - 2000 Hz Drake MultibodyPlant + SAP Contact Solver
                                 - 2000 Hz In-memory joint PD feedback loop
                                 - 500 Hz MRT policy follower & Pinocchio inverse dynamics
                                 - Meshcat WebSocket server

Cores 10 - 11 (2 threads, 0xc00):[Node 3 & OS: Desktop, GUI & Browser]
                                 - base_velocity_controller_gui (25 Hz Tkinter)
                                 - Web browser / WebGL Meshcat visualizer
                                 - Linux display server, Xorg/Wayland, background daemons
```

### 3.2 Launch File Configuration (`drake_mrt_sim.launch.py`)
In `src/wb_humanoid_mpc/robot_models/unitree_g1/g1_centroidal_mpc/launch/drake_mrt_sim.launch.py`:

```python
# Node 1: OCS2 Policy Publisher Node (Cores 0-4)
mpc_node = Node(
    package="humanoid_centroidal_mpc_ros2",
    executable="humanoid_centroidal_mpc_sqp_node",
    name="humanoid_centroidal_mpc_sqp_node",
    prefix=["taskset -c 0-4"],
    output="screen",
    emulate_tty=True,
    arguments=[
        LaunchConfiguration("robot_name"),
        LaunchConfiguration("config_name"),
        LaunchConfiguration("target_command_file"),
        LaunchConfiguration("description_name"),
        LaunchConfiguration("target_gait_file"),
    ],
)

# Node 2: Drake MRT Follower & Sim Node (Cores 5-9)
drake_mrt_node = Node(
    package="humanoid_drake_sim",
    executable="drake_mrt_sim_node",
    name="drake_mrt_sim_node",
    prefix=["taskset -c 5-9"],
    output="screen",
    emulate_tty=True,
    arguments=[
        LaunchConfiguration("robot_name"),
        LaunchConfiguration("config_name"),
        LaunchConfiguration("target_command_file"),
        LaunchConfiguration("description_name"),
        LaunchConfiguration("target_gait_file"),
    ],
    parameters=[{
        "sim_dt": 0.0005,
        "control_rate_hz": 500.0,
        "joint_damping": 10.0,
        "ground_static_friction": 2.0,
        "ground_dynamic_friction": 1.5,
        "ground_dissipation": 2.0,
        "ground_point_stiffness": 5.0e4,
        "force_mujoco_contact_flags": True,
        "reset_on_fall": True,
        "fall_z_threshold": 0.2,
        "reset_cooldown_s": 1.0,
        "enable_meshcat": True,          # Enabled by default per R5
        "meshcat_host": "*",
        "meshcat_publish_rate_hz": 30.0,
        "verbose": True,
    }],
)
```

### 3.3 Runtime Core Verification
Verify core pinning while the simulation is running:
```bash
# Verify Node 1 is pinned to cores 0-4 (mask 1f)
taskset -cp $(pgrep -f humanoid_centroidal_mpc_sqp_node)
# pid <PID>'s current affinity list: 0-4

# Verify Node 2 is pinned to cores 5-9 (mask 3e0)
taskset -cp $(pgrep -f drake_mrt_sim_node)
# pid <PID>'s current affinity list: 5-9
```

---

## 4. Startup & Reset Synchronization Protocol (Squat Prevention)

### 4.1 Root Cause of the Startup Squat Phenomenon
In `TargetTrajectoriesCalculatorBase.cpp:114-127`:
```cpp
vector4_t TargetTrajectoriesCalculatorBase::filterAndTransformVelCommandToLocal(
    const vector4_t& commandedVelLocal, const scalar_t& currentEulerZ, scalar_t filterAlpha) const {
  static vector4_t commVelFiltered = vector4_t::Zero();

  commVelFiltered = commVelFiltered * filterAlpha + commandedVelLocal * (1 - filterAlpha);

  vector4_t globalTargetVel = commVelFiltered;
  globalTargetVel(0) = std::cos(currentEulerZ) * commVelFiltered[0] - std::sin(currentEulerZ) * commVelFiltered[1];
  globalTargetVel(1) = std::sin(currentEulerZ) * commVelFiltered[0] + std::cos(currentEulerZ) * commVelFiltered[1];
  return globalTargetVel;
}
```

- **The Static Filter**: `commVelFiltered` is declared as a `static` local variable, initialized to `(0, 0, 0, 0)`.
- Index 2 represents pelvis height ($z_{\text{nominal}} = 0.7925$ m).
- With $\alpha = 0.8$, the filtered height ramps slowly from 0:
  - Step 1: $0.2 \times 0.7925 = 0.1585$ m
  - Step 2: $0.8 \times 0.1585 + 0.2 \times 0.7925 = 0.2853$ m
  - Step 3: $0.3867$ m ... Step 10: $0.707$ m ... Step 25: $0.790$ m
- **Consequence**: If physics begins stepping with a cold filter, the first 15 MPC policies command the robot to squat down to $0.16–0.50$ m. The robot collapses into the ground and triggers the fall reset threshold ($z < 0.20$ m).

---

### 4.2 The Complete 4-Phase Synchronization Protocol

```text
Sequence Diagram: Startup & Reset Synchronization Protocol
========================================================================================
Node 2 (drake_mrt_sim_node)                          Node 1 (humanoid_centroidal_mpc_sqp_node)
---------------------------                          -----------------------------------------
[Phase 1: Posture Hold]
1. Initialize Drake plant at z_spawn = 0.7872 m
   Apply holding joint PD (Kp=1200, Kd=10, tau_g)
   Do NOT step physics yet (sim thread held).

[Phase 2: Service Reset]
2. Call /g1/mpc_reset service ---------------------> Lock resetMutex_
   Wait for res->done == true <--------------------- SqpMpc::reset()
                                                     Set resetRequestedEver_ = true
                                                     Assign nominal TargetTrajectories

[Phase 3: Observation Warm-up Loop]
3. Loop 25-30 cycles (or until planned z >= 0.78m):
   Publish /g1/mpc_observation (t=0, x_init) ------> mpcObservationCallback()
                                                     preSolverRun() executes filter:
                                                     commVelFiltered[2] -> 0.7925 m
   Receive /g1/mpc_policy <------------------------- Publishes warm policy (z_ref >= 0.78 m)
   mrt.updatePolicy()
   Inspect policy planned height >= 0.78 m

[Phase 4: Physics Stepping Release]
4. Start 2000 Hz physics simulation thread
   Start 500 Hz closed-loop MRT follower loop:
     Evaluate policy at t_sim + 0.005
     Compute Pinocchio tau_ff
     DrakeSimInterface in-memory PD
     Publish /g1/mpc_observation (t_sim, x_meas) --> Solve dynamic walking/standing
     Update MRT policy <---------------------------- Publishes live policy
========================================================================================
```

---

## 5. Drake Physics, Actuation & Ground Contact Mechanics

### 5.1 2000 Hz SAP Contact Solver Configuration
- **Step Size**: $dt = 0.0005$ s (2000 Hz).
- **Contact Approximation**: `drake::multibody::DiscreteContactApproximation::kSap`.
- **Contact Model**: `drake::multibody::ContactModel::kPointContactOnly`.
- **Physical Parameters**:
  - `groundPointStiffness`: $5.0 \times 10^4$ N/m (Prevents penetration while avoiding numerical stiffness).
  - `groundDissipation`: $2.0$ s/m (Normal velocity damping during contact impact).
  - `groundStaticFriction`: $2.0$ & `groundDynamicFriction`: $1.5$ (Provides high tangential traction matching MuJoCo).
  - `jointDamping`: $10.0$ N$\cdot$s/rad (Applied to all 29 revolute joints).

### 5.2 Foot Contact Geometry & Collision Filtering
- **Collision Stripping**: In `g1_29dof_drake.urdf`, all upper-body mesh collisions are stripped. Only 4 contact spheres ($r = 5$ mm) per foot (8 spheres total) are retained on `left_ankle_roll_link` and `right_ankle_roll_link`.
- **Self-Collision Exclusion**: `DrakeSimInterface::buildPlant()` enforces `CollisionFilterDeclaration::ExcludeWithin` across the entire robot model instance, eliminating all 703 internal collision pairs.

### 5.3 Calibrated Pelvis Spawn Height ($z_{\text{spawn}} = 0.7872$ m)
- Nominal standing pelvis height in `task.info` is $0.7925$ m.
- Forward kinematics reveals that at $0.7925$ m, the foot spheres float **$5.3$ mm** above the ground ($z = +0.0053$ m).
- Calibrated spawn height:
  $$z_{\text{spawn}} = 0.7925 - 0.0053 = 0.7872\text{ m}$$
- At $0.7872$ m, foot spheres rest tangentially on the ground at $t = 0$ with zero drop shock.

### 5.4 In-Memory Joint PD Loop & Strict Torque Clamping
At each 2000 Hz step in `DrakeSimInterface::simulationStep()`:
$$\tau_i = \text{clamp}\left( \tau_{\text{ff},i} + K_{p,i}(q_{\text{des},i} - q_i) + K_{d,i}(\dot{q}_{\text{des},i} - \dot{q}_i), \; -\tau_{\text{max},i}, \; \tau_{\text{max},i} \right)$$
- Gains: $K_p = 1200.0$ N$\cdot$m/rad, $K_d = 10.0$ N$\cdot$m$\cdot$s/rad.
- Clamping limits: $\pm 88$ N$\cdot$m for hips, $\pm 139$ N$\cdot$m for knees, $\pm 50$ N$\cdot$m for ankles.

### 5.5 Pinocchio `computeJointTorques` at 500 Hz
In the MRT follower control loop, inverse dynamics computes feedforward torques $\tau_{\text{ff}}$:
1. Inertia matrix $M(q)$ via Composite Rigid Body Algorithm (CRBA).
2. Non-linear effects $h(q, \dot{q}) = C(q, \dot{q})\dot{q} + g(q)$.
3. Contact Jacobians $J_L, J_R$ for foot contact frames.
4. Contact wrench projection: $\tau_{\text{ext}} = J_L^T w_L + J_R^T w_R$.
5. Solves unactuated floating base acceleration:
   $$M_{bb} \ddot{q}_b + M_{bj} \ddot{q}_j + h_b - \tau_{\text{ext},b} = 0 \implies \ddot{q}_b = -M_{bb}^{-1} (M_{bj} \ddot{q}_j + h_b - \tau_{\text{ext},b})$$
6. Actuated joint torques:
   $$\tau_{\text{ff}} = M_{jb} \ddot{q}_b + M_{jj} \ddot{q}_j + h_j - \tau_{\text{ext},j}$$

---

## 6. Comprehensive Inventory of All Executables & Libraries in `humanoid_drake_sim`

The package builds 1 library (`drake_sim_interface`) and 10 standalone executables:

```text
Diagnostic Verification Progression:
  Level 0: [drake_urdf_interface_smoke_test]  --> URDF parsing, 29 actuators, 10-step finite state
  Stage 1: [kinematic_replay_test]            --> MPC plan replay, Euler angles, joint signs (Physics OFF)
  Stage 2: [static_equilibrium_check]         --> StaticEquilibriumProblem gate, contact geometry, CoM polygon
  Stage 3: [gravity_comp_hold_test]           --> Analytical tau_g + PD posture hold + passive damping (Physics ON)
           [drake_sap_stand_test]             --> Standalone Drake SAP implicit joint PD constraint
  Stage 4: [policy_playback_test]             --> Single-solve optimal trajectory replay, force sanity check
  Stage 5: [threaded_closed_loop_test]        --> Multithreaded closed loop, solver overruns vs physics pacing
  Production: [drake_mrt_sim_node]            --> Production 2-Node Drake MRT Policy Follower & Simulator
  Legacy:     [drake_sim_node]                --> Legacy standalone simulator (/drake/joint_states)
              [policy_to_command_node]        --> Legacy bridge node (/drake/joint_torques)
```

---

### Target 1: `drake_sim_interface` (C++ Library)
- **Source**: `src/DrakeSimInterface.cpp`, `include/humanoid_drake_sim/DrakeSimInterface.h`.
- **Purpose**: Reusable C++ hardware interface implementing `model::RobotHWInterfaceBase` for Drake. Houses the 2000 Hz discrete SAP simulation loop, in-memory PD feedback, torque clamping, self-collision exclusion, and decoupled 60 Hz Meshcat visualizer.

### Target 2: `drake_mrt_sim_node` (Production Drake MRT Follower & Sim)
- **Source**: `src/drake_mrt_sim_node.cpp`.
- **Purpose**: Production simulator node launched by `drake_mrt_sim.launch.py`. Uses `ocs2::MRT_ROS_Interface` to follow `/g1/mpc_policy`, publishes `/g1/mpc_observation`, evaluates Pinocchio inverse dynamics at 500 Hz, steps Drake physics at 2000 Hz, and serves Meshcat at port 7000.
- **Execution Command**:
  ```bash
  ros2 launch g1_centroidal_mpc drake_mrt_sim.launch.py
  ```

### Target 3: `drake_urdf_interface_smoke_test` (Level 0 Smoke Test)
- **Source**: `src/drake_urdf_interface_smoke_test.cpp`.
- **Purpose**: Validates URDF parsing, discovers all 29 revolute joints as actuators, steps physics forward by 10 intervals (5 ms), and verifies all states are finite numbers (no `NaN` or `Inf`).
- **Execution Command**:
  ```bash
  ros2 run humanoid_drake_sim drake_urdf_interface_smoke_test \
    $(ros2 pkg prefix g1_description)/share/g1_description/urdf/g1_29dof_drake.urdf --meshcat
  ```

### Target 4: `kinematic_replay_test` (Stage 1 Diagnostic)
- **Source**: `src/kinematic_replay_test.cpp`.
- **Purpose**: Replays an OCS2 planned trajectory kinematically in Drake (`dt = 0.0`, physics disabled) to verify coordinate frame alignment, Euler ZYX order, and joint sign conventions.
- **Execution Command**:
  ```bash
  ros2 run humanoid_drake_sim kinematic_replay_test g1 \
    $(ros2 pkg prefix g1_centroidal_mpc)/share/g1_centroidal_mpc/config/mpc/task.info \
    $(ros2 pkg prefix g1_centroidal_mpc)/share/g1_centroidal_mpc/config/command/reference.info \
    $(ros2 pkg prefix g1_description)/share/g1_description/urdf/g1_29dof.urdf \
    $(ros2 pkg prefix g1_centroidal_mpc)/share/g1_centroidal_mpc/config/mpc/gait.info \
    $(ros2 pkg prefix g1_description)/share/g1_description/urdf/g1_29dof_drake.xml 3 0.5
  ```

### Target 5: `static_equilibrium_check` (Stage 2 Gate)
- **Source**: `src/static_equilibrium_check.cpp`.
- **Purpose**: Solves a `drake::multibody::StaticEquilibriumProblem` over an `AutoDiffXd` plant to verify whether static standing configuration and contact reaction forces exist without stepping physics.
- **Execution Command**:
  ```bash
  ros2 run humanoid_drake_sim static_equilibrium_check \
    $(ros2 pkg prefix g1_description)/share/g1_description/urdf/g1_29dof_drake.xml 0.7872
  ```

### Target 6: `gravity_comp_hold_test` (Stage 3 Diagnostic)
- **Source**: `src/gravity_comp_hold_test.cpp`.
- **Purpose**: Evaluates pure Drake dynamics under analytical gravity compensation plus PD posture hold ($\tau = -\tau_g(q) + K_p(q_{\text{nom}}-q) - K_d \dot{q}$) without MPC. Proves contact parameter stability.
- **Execution Command**:
  ```bash
  ros2 run humanoid_drake_sim gravity_comp_hold_test \
    $(ros2 pkg prefix g1_description)/share/g1_description/urdf/g1_29dof_drake.xml 10.0 0.7872
  ```

### Target 7: `drake_sap_stand_test` (Benchmark Diagnostic)
- **Source**: `src/drake_sap_stand_test.cpp`.
- **Purpose**: Diagnostic benchmark configuring per-joint PD gains directly on actuators using Drake's internal `SapPdControllerConstraint` (implicit spring-damper contact).
- **Execution Command**:
  ```bash
  ros2 run humanoid_drake_sim drake_sap_stand_test
  ```

### Target 8: `policy_playback_test` (Stage 4 Diagnostic)
- **Source**: `src/policy_playback_test.cpp`.
- **Purpose**: Solves OCS2 SQP once, extracts planned trajectories, and replays them open-loop in Drake physics. Validates contact force normalization ($F_z \approx 172$ N per foot for 35 kg robot).
- **Execution Command**:
  ```bash
  ros2 run humanoid_drake_sim policy_playback_test g1 \
    $(ros2 pkg prefix g1_centroidal_mpc)/share/g1_centroidal_mpc/config/mpc/task.info \
    $(ros2 pkg prefix g1_centroidal_mpc)/share/g1_centroidal_mpc/config/command/reference.info \
    $(ros2 pkg prefix g1_description)/share/g1_description/urdf/g1_29dof.urdf \
    $(ros2 pkg prefix g1_centroidal_mpc)/share/g1_centroidal_mpc/config/mpc/gait.info \
    $(ros2 pkg prefix g1_description)/share/g1_description/urdf/g1_29dof_drake.xml 3.0
  ```

### Target 9: `threaded_closed_loop_test` (Stage 5 Diagnostic)
- **Source**: `src/threaded_closed_loop_test.cpp`.
- **Purpose**: Threaded closed-loop diagnostic isolating SQP solver compute spikes from physics pacing. Confirms that solver overruns do not stall 2000 Hz physics integration.
- **Execution Command**:
  ```bash
  ros2 run humanoid_drake_sim threaded_closed_loop_test g1 \
    $(ros2 pkg prefix g1_centroidal_mpc)/share/g1_centroidal_mpc/config/mpc/task.info \
    $(ros2 pkg prefix g1_centroidal_mpc)/share/g1_centroidal_mpc/config/command/reference.info \
    $(ros2 pkg prefix g1_description)/share/g1_description/urdf/g1_29dof.urdf \
    $(ros2 pkg prefix g1_centroidal_mpc)/share/g1_centroidal_mpc/config/mpc/gait.info \
    $(ros2 pkg prefix g1_description)/share/g1_description/urdf/g1_29dof_drake.xml
  ```

### Target 10: `drake_sim_node` (Legacy Standalone Simulator)
- **Source**: `src/drake_sim_node.cpp`.
- **Purpose**: Historical standalone simulator communicating over ad-hoc ROS 2 topics (`/drake/joint_states`, `/drake/joint_torques`).
- **Execution Command**:
  ```bash
  ros2 run humanoid_drake_sim drake_sim_node
  ```

### Target 11: `policy_to_command_node` (Legacy Bridge Node)
- **Source**: `src/policy_to_command_node.cpp`.
- **Purpose**: Historical controller bridge subscribing to `/drake/joint_states` and streaming raw joint torques over `/drake/joint_torques`.
- **Execution Command**:
  ```bash
  ros2 run humanoid_drake_sim policy_to_command_node
  ```

---

## 7. Teleoperation, Walking Control & Meshcat Visualization

### 7.1 GUI Velocity Command Pipeline
- **Publisher**: `base_velocity_controller_gui.py` (`remote_control` package, 25 Hz).
- **Topic**: `/humanoid/walking_velocity_command` (`humanoid_mpc_msgs/msg/WalkingVelocityCommand`, Best Effort QoS).
- **Velocity Scaling**:
  - $v_{x,\text{scaled}} = \text{linear\_velocity\_x} \times 2.4$ m/s
  - $v_{y,\text{scaled}} = \text{linear\_velocity\_y} \times 1.2$ m/s
  - $\omega_{z,\text{scaled}} = \text{angular\_velocity\_z} \times 1.0$ rad/s
  - Default Pelvis Height: $0.7925$ m (corresponds to GUI slider value $\approx 74$).

### 7.2 Locomotion State Machine & Instant Stance Breakout
In `ProceduralMpcMotionManager.cpp`:
- Gait mode progression: `stance` (0), `slow_walk` (1), `walk` (2), `trot` (5).
- In `stance` mode, `linVelErrorThresh = 10.0` m/s.
- Condition: $|v_{\text{base}}| > v_{\text{max}} - v_{\text{thresh}} = 0.10 - 10.0 = -9.9$ m/s is identically true!
- Any command $|v_{x,\text{cmd}}| > 0.10$ m/s instantly transitions the robot out of `stance` into `slow_walk` and `walk` without lag.

### 7.3 Exact CLI Commands for Forward & Backward Walking
- **Forward Walking Command** ($v_x = +0.40$, scaled $0.96$ m/s):
  ```bash
  ros2 topic pub --once /humanoid/walking_velocity_command humanoid_mpc_msgs/msg/WalkingVelocityCommand \
    "{linear_velocity_x: 0.4, linear_velocity_y: 0.0, angular_velocity_z: 0.0, desired_pelvis_height: 0.7925}"
  ```
  Criterion: Net forward displacement $\Delta x > +0.30$ m over 5–10 s of simulation time with zero falls.

- **Backward Walking Command** ($v_x = -0.40$, scaled $-0.96$ m/s):
  ```bash
  ros2 topic pub --once /humanoid/walking_velocity_command humanoid_mpc_msgs/msg/WalkingVelocityCommand \
    "{linear_velocity_x: -0.4, linear_velocity_y: 0.0, angular_velocity_z: 0.0, desired_pelvis_height: 0.7925}"
  ```
  Criterion: Net backward displacement $\Delta x < -0.20$ m over 5–10 s of simulation time with zero falls.

### 7.4 Meshcat Visualization & Browser Verification
- **Binding**: Meshcat server binds to `http://localhost:7000`.
- **Decoupled Architecture**: Zero-timestep diagram updated at 60 Hz; network delays never block 2000 Hz physics.
- **Scene Hierarchy**:
  - `/robot`: Root visualizer path.
  - `/robot/ground_visual`: Grey ground slab ($20 \times 20 \times 0.2$ m).
  - `/robot/g1/pelvis`: Robot floating base coordinate frame.
- **Server Status Inspection**:
  ```text
  Meshcat server scene: scene=1 ground=1 g1_root=1 clients=...
  ```
- **Verification Commands**:
  ```bash
  curl -s -I http://localhost:7000 | grep -E "HTTP/1.1 200|OK"
  ```

---

## 8. Verification Methodology, Anti-Freeze Rules & Acceptance Criteria

### 8.1 Mandatory Build Command
**NEVER** run raw `colcon build` directly. Always build from the package root:
```bash
cd /home/mov/Documents/humanoid_mpc_ws/src/wb_humanoid_mpc && make build-all PARALLEL_JOBS=1
```

### 8.2 Single Simulation Instance Anti-Freeze Rule
The host machine has 12 CPU threads. At most **ONE** simulation instance (`drake_mrt_sim_node` or `ros2 launch`) may execute at any time. Running concurrent instances will freeze the machine.

### 8.3 Acceptance Criteria Checklist
- [ ] Zero modifications to `humanoid_centroidal_mpc`, `humanoid_centroidal_mpc_ros2`, `humanoid_common_mpc`, `humanoid_common_mpc_ros2`, `humanoid_wb_mpc`, `mujoco_sim_interface`, `remote_control`.
- [ ] `make build-all PARALLEL_JOBS=1` exits with code 0.
- [ ] `ros2 topic info /g1/mpc_policy` and `/g1/mpc_observation` confirm active publisher/subscriber connections between Node 1 and Node 2.
- [ ] Process inspection confirms MPC solver is pinned to cores `0-4` and Drake MRT sim is pinned to cores `5-9`.
- [ ] Unitree G1 stands stably upright ($p_z \in [0.65, 0.90]$ m, $|\text{pitch}| < 0.35$, $|\text{roll}| < 0.35$, $\text{falls} = 0$) for $\ge 15$ seconds.
- [ ] Forward command ($v_x = 0.4$) produces $\Delta x > 0.30$ m without falls.
- [ ] Backward command ($v_x = -0.4$) produces $\Delta x < -0.20$ m without falls.
- [ ] Meshcat server at `http://localhost:7000` reports `scene=1 ground=1 g1_root=1` and renders standing/walking robot cleanly in browser.

