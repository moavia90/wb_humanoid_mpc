from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from humanoid_common_mpc_ros2.mpc_launch_config import MPCLaunchConfig


def generate_launch_description():
    cfg = MPCLaunchConfig(
        mpc_lib_pkg="humanoid_centroidal_mpc",
        mpc_config_pkg="g1_centroidal_mpc",
        mpc_model_pkg="g1_description",
        urdf_rel_path="/urdf/g1_29dof_drake.urdf",
        xml_rel_path="/urdf/g1_29dof.xml",
        robot_name="g1",
        solver="sqp",
        enable_debug=False,
    )

    launch_gui_arg = DeclareLaunchArgument(
        "launch_gui",
        default_value="true",
        description="Whether to launch base velocity controller GUI",
    )

    enable_meshcat_arg = DeclareLaunchArgument(
        "enable_meshcat",
        default_value="true",
        description="Whether to enable Drake Meshcat visualizer",
    )

    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="false",
        description="Whether to launch RViz (default: false, Meshcat is primary)",
    )

    # 1. Standalone OCS2 Policy Publisher Node (humanoid_centroidal_mpc_sqp_node)
    # Pinned to CPU cores 0-4 (mask 0x1f).
    # Publishes /g1/mpc_policy, subscribes to /g1/mpc_observation & /humanoid/walking_velocity_command_clamped,
    # and provides /g1/mpc_reset service.
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
        remappings=[
            (
                "humanoid/walking_velocity_command",
                "/humanoid/walking_velocity_command_clamped",
            ),
        ],
    )

    # 2. Drake MRT Follower & Physics Sim Node (drake_mrt_sim_node)
    # Pinned to CPU cores 5-9 (mask 0x3e0).
    # Subscribes to /g1/mpc_policy, publishes /g1/mpc_observation, calls /g1/mpc_reset.
    # Intercepts /humanoid/walking_velocity_command, applies safe physical velocity scaling/clamping
    # and fall-recovery zero-velocity hold, and forwards to /humanoid/walking_velocity_command_clamped.
    # Evaluates MRT policy & Pinocchio inverse dynamics at 500 Hz, steps Drake physics at 2000 Hz.
    # Renders 3D visuals to Meshcat (localhost:7000) when enable_meshcat is true.
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
        parameters=[
            {
                # Match the existing MuJoCo backend's physics rate (2000 Hz).
                "sim_dt": 0.0005,
                # Match MuJoCo baseline MRT rate (500 Hz).
                "control_rate_hz": 500.0,
                # Passive joint damping (matching MuJoCo defaultJointDamping = 10.0).
                "joint_damping": 10.0,
                # Ground contact parameters providing strong traction and stable SAP compliance.
                "ground_static_friction": 2.0,
                "ground_dynamic_friction": 1.5,
                "ground_dissipation": 2.0,
                "ground_point_stiffness": 5.0e4,
                # Keep parity with the current MuJoCo implementation, whose
                # RobotState currently reports both contacts as true.
                "force_mujoco_contact_flags": True,
                # Same fall-reset behavior as MujocoSimInterface.
                "reset_on_fall": True,
                "fall_z_threshold": 0.2,
                "reset_cooldown_s": 1.0,
                # Safe physical velocity command envelope (Root Cause 3 & A fix).
                "max_forward_vel_x": 0.65,
                "max_backward_vel_x": 0.48,
                "max_lateral_vel_y": 0.30,
                "max_yaw_rate_z": 0.32,
                "min_pelvis_height": 0.68,
                "max_pelvis_height": 0.84,
                # Meshcat renders state snapshots from the main thread at meshcat_publish_rate_hz.
                # Default is True per Requirement R5.
                "enable_meshcat": ParameterValue(
                    LaunchConfiguration("enable_meshcat"), value_type=bool
                ),
                # Drake's documented all-interfaces wildcard. The URL it
                # prints uses localhost, while remote clients can still use
                # this machine's IP address.
                "meshcat_host": "*",
                "meshcat_publish_rate_hz": 30.0,
                "verbose": True,
            }
        ],
    )

    # 3. Base Velocity Controller GUI (base_velocity_controller_gui)
    # Unpinned: runs on remaining CPU cores 10-11 along with desktop UI and Meshcat client.
    gui_node = Node(
        package="remote_control",
        executable="base_velocity_controller_gui",
        name="base_velocity_controller_gui",
        output="screen",
        condition=IfCondition(LaunchConfiguration("launch_gui")),
    )

    # 4. Optional RViz visualization node
    # Unpinned: runs on cores 10-11 when rviz:=true (Meshcat is default visualizer).
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rvizconfig")],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            # Launch arguments
            launch_gui_arg,
            enable_meshcat_arg,
            rviz_arg,
            cfg.declare_robot_name,
            cfg.declare_config_file,
            cfg.declare_target_command_file,
            cfg.declare_gait_command_file,
            cfg.declare_urdf_path,
            cfg.declare_rviz_config_path,
            # Nodes
            mpc_node,
            drake_mrt_node,
            gui_node,
            rviz_node,
        ]
    )
