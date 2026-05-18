###############################################################################
# drake_sim.launch.py
#
# Launches the Drake simulator, centroidal MPC, and the Drake policy bridge.
#
# Startup order:
#   t=0s  Drake, centroidal MPC, and the bridge all start.
#         Drake pins the nominal reset pose until the bridge receives the
#         first MPC policy and explicitly releases the pin.
#         The bridge waits for Drake state and /g1/mpc_reset, then resets MPC.
###############################################################################

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
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

    drake_sim_node = Node(
        package="humanoid_drake_sim",
        executable="drake_sim_node",
        name="drake_sim_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "urdf_path": (
                get_package_share_directory("g1_description")
                + "/urdf/g1_29dof_drake.xml"
            ),
            "sim_dt": 0.001,
            "max_torque": 180.0,
            "torque_timeout_s": 0.2,
            "pin_base_until_first_torque": True,
            "startup_release_torque_threshold": -1.0,
        }],
    )

    bridge_node = Node(
        package="humanoid_drake_sim",
        executable="policy_to_command_node",
        name="policy_to_command_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "control_rate_hz": 500.0,
            "obs_decimation": 5,
            "reset_retry_period_s": 5.0,
            "min_reset_time_s": 0.02,
            "reset_valid_samples": 3,
            "publish_hold_torques": False,
            "use_contact_mode_for_observation": False,
            "fixed_observation_mode": 3,
            "mpc_feedforward_scale": 0.0,
            "max_policy_joint_velocity": 6.0,
        }],
    )

    standard_nodes = [
        cfg.robot_state_publisher_node,
        cfg.base_velocity_controller_gui_node,
        cfg.rviz_node,
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
        + [drake_sim_node, cfg.mpc_node, bridge_node]
        + standard_nodes
    )
