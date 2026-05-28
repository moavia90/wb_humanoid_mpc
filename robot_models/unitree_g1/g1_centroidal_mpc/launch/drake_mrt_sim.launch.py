from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from humanoid_common_mpc_ros2.mpc_launch_config import MPCLaunchConfig


def generate_launch_description():
    cfg = MPCLaunchConfig(
        mpc_lib_pkg="humanoid_centroidal_mpc",
        mpc_config_pkg="g1_centroidal_mpc",
        mpc_model_pkg="g1_description",
        urdf_rel_path="/urdf/g1_29dof.urdf",
        xml_rel_path="/urdf/g1_29dof.xml",
        robot_name="g1",
        solver="sqp",
        enable_debug=False,
    )

    drake_model_path = (
        get_package_share_directory("g1_description")
        + "/urdf/g1_29dof_drake.xml"
    )

    drake_mrt_node = Node(
        package="humanoid_drake_sim",
        executable="drake_mrt_sim_node",
        name="drake_mrt_sim_node",
        output="screen",
        emulate_tty=True,

        # Args expected by drake_mrt_sim_node.cpp:
        #
        #   program_args[1] = robot_name
        #   program_args[2] = task_file
        #   program_args[3] = reference_file
        #   program_args[4] = urdf_file
        #   program_args[5] = gait_file
        #   program_args[6] = drake_model_file
        arguments=[
            LaunchConfiguration("robot_name"),
            LaunchConfiguration("config_name"),
            LaunchConfiguration("target_command_file"),
            LaunchConfiguration("description_name"),
            LaunchConfiguration("target_gait_file"),
            drake_model_path,
        ],

        parameters=[{
            # ------------------------------------------------------------------
            # [3] MuJoCo-equivalent physics timestep.
            #
            # MuJoCo uses dt=0.0005. Keep this in Drake too.
            # The node automatically runs multiple Drake substeps per control
            # tick if control_rate_hz is slower than 1 / sim_dt.
            # ------------------------------------------------------------------
            "sim_dt": 0.0005,
            "control_rate_hz": 250.0,

            # ------------------------------------------------------------------
            # Actuator safety.
            #
            # Drake ignores some MJCF actuator limit attributes, so the C++ node
            # clamps torques manually using min(max_torque, joint_specific_limit).
            # ------------------------------------------------------------------
            "max_torque": 180.0,

            # -1.0 means compute initial pelvis height from foot pose.
            # Try manual values only after the reset/damping/direct-torque version
            # is running: 0.76, 0.74, 0.72.
            "initial_height": -1.0,

            # ------------------------------------------------------------------
            # Startup pin.
            #
            # Keep 2.0 initially. It is Drake-only warm-start protection.
            # Once the robot stands/walks, reduce to 0.5, then 0.0.
            # ------------------------------------------------------------------
            "startup_pin_duration_s": 2.0,

            # ------------------------------------------------------------------
            # [1] Auto reset on fall, MuJoCo-style.
            #
            # When fallen, Drake resets base/joints/velocities, zeros torques,
            # pins again, and waits through cooldown.
            # ------------------------------------------------------------------
            "reset_on_fall": True,
            "reset_cooldown_s": 1.0,
            "fall_z_threshold": 0.25,
            "fall_angle_threshold_rad": 0.90,

            # ------------------------------------------------------------------
            # [2] Passive damping equivalent to MuJoCo dof_damping = 10.
            # ------------------------------------------------------------------
            "passive_joint_damping": 10.0,

            # ------------------------------------------------------------------
            # [4] MuJoCo-equivalent contact flags.
            #
            # MuJoCo code currently sets both foot contacts true. This makes
            # Drake RobotState match that behavior. Real Drake contact forces
            # are still printed in debug logs.
            # ------------------------------------------------------------------
            "force_mujoco_contact_flags": True,

            # ------------------------------------------------------------------
            # [5] Direct MuJoCo-style torque mode.
            #
            # This applies:
            #   JointAction::getTotalFeedbackTorque(q, qd)
            # directly to Drake actuators, plus passive damping and clamps.
            # No PD/MRT blending.
            # ------------------------------------------------------------------
            "direct_mujoco_torque_mode": True,

            # ------------------------------------------------------------------
            # Drake/SAP control ladder.
            #
            # "sap_pd_stand": verify Drake can hold G1 upright without MRT torque.
            # "mrt_walk":     SAP PD hold, then fade SAP PD out while MRT torque
            #                 ramps in for walking experiments.
            # "mrt_direct":   old behavior: direct MRT torque after startup pin.
            # ------------------------------------------------------------------
            "control_mode": "sap_pd_stand",
            "enable_sap_pd_controller": True,
            "sap_pd_hold_duration_s": 2.0,
            "sap_pd_blend_duration_s": 3.0,
            "mrt_torque_scale": 1.0,
            "max_torque_rate": 1200.0,

            # ------------------------------------------------------------------
            # [6] Manual compensation for Drake-ignored MJCF contact/friction
            # fields. This is not exact MuJoCo behavior; it is a practical
            # approximation.
            # ------------------------------------------------------------------
            "ground_static_friction": 0.9,
            "ground_dynamic_friction": 0.7,
            "ground_dissipation": 2.0,
            "ground_point_stiffness": 5.0e4,
            "ground_hydro_modulus": 5.0e4,
            "ground_hydro_margin": 0.05,

            # Meshcat.
            # Use "0.0.0.0" for Docker/external browser access.
            "meshcat_host": "0.0.0.0",

            # Debug.
            "debug_publish_rate_hz": 50.0,
            "debug_log_period_s": 0.25,
            "debug_top_joints": 8,
            "debug_verbose": True,
        }],
    )

    # Keep GUI velocity command. Keep RViz and robot_state_publisher disabled
    # during timing/contact debugging.
    standard_nodes = [
        cfg.base_velocity_controller_gui_node,

        # Disabled intentionally:
        # cfg.robot_state_publisher_node,
        # cfg.rviz_node,
    ]

    return LaunchDescription(
        [
            cfg.declare_robot_name,
            cfg.declare_config_file,
            cfg.declare_target_command_file,
            cfg.declare_gait_command_file,
            cfg.declare_urdf_path,
            cfg.declare_rviz_config_path,
        ]
        + [drake_mrt_node]
        + standard_nodes
    )
