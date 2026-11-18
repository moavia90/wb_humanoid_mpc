#!/usr/bin/env python3
"""
Milestone 1 Adversarial Challenge & Verification Test Suite.

Tests launch file evaluation, parameter coercion, condition branching,
and taskset CPU affinity bitmask correctness for drake_mrt_sim.launch.py.
"""

import importlib.util
import os
import pytest
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.conditions import InvalidConditionExpressionError
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters


LAUNCH_FILE_PATH = Path(
    "/home/mov/Documents/humanoid_mpc_ws/src/wb_humanoid_mpc/"
    "robot_models/unitree_g1/g1_centroidal_mpc/launch/drake_mrt_sim.launch.py"
)


def load_launch_module():
    """Dynamically loads the drake_mrt_sim.launch.py module."""
    assert LAUNCH_FILE_PATH.exists(), f"Launch file not found at {LAUNCH_FILE_PATH}"
    spec = importlib.util.spec_from_file_location("launch_module", LAUNCH_FILE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_core_range(range_str: str) -> set:
    """Parses a core range string (e.g. '0-4') into a set of CPU integer IDs."""
    if "-" in range_str:
        start, end = map(int, range_str.split("-"))
        return set(range(start, end + 1))
    return {int(range_str)}


def cores_to_mask(cores: set) -> int:
    """Computes bitmask for a set of CPU core IDs."""
    mask = 0
    for core in cores:
        mask |= (1 << core)
    return mask


def setup_context_with_args(launch_description, arg_dict: dict) -> LaunchContext:
    """Initializes a LaunchContext with the given launch arguments and visits all DeclareLaunchArgument actions."""
    ctx = LaunchContext()
    for k, v in arg_dict.items():
        ctx.launch_configurations[k] = str(v)
    for entity in launch_description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(ctx)
    return ctx


class TestMilestone1LaunchVerification:
    """Adversarial stress and correctness test suite for Milestone 1."""

    def test_launch_description_entities(self):
        """Verifies that the launch description creates expected entities."""
        mod = load_launch_module()
        ld = mod.generate_launch_description()
        assert ld is not None, "Launch description must not be None"
        assert len(ld.entities) == 13, f"Expected 13 entities, got {len(ld.entities)}"

        declared_args = [
            e.name for e in ld.entities if isinstance(e, DeclareLaunchArgument)
        ]
        expected_args = [
            "launch_gui",
            "enable_meshcat",
            "rviz",
            "robot_name",
            "config_name",
            "target_command_file",
            "target_gait_file",
            "description_name",
            "rvizconfig",
        ]
        for expected in expected_args:
            assert expected in declared_args, f"Missing declared launch argument: {expected}"

    @pytest.mark.parametrize(
        "enable_meshcat,launch_gui,rviz,expected_meshcat_bool,expect_gui_active,expect_rviz_active",
        [
            # All 8 boolean combinations (lowercase)
            ("true", "true", "true", True, True, True),
            ("true", "true", "false", True, True, False),
            ("true", "false", "true", True, False, True),
            ("true", "false", "false", True, False, False),
            ("false", "true", "true", False, True, True),
            ("false", "true", "false", False, True, False),
            ("false", "false", "true", False, False, True),
            ("false", "false", "false", False, False, False),
            # Capitalized strings
            ("True", "True", "True", True, True, True),
            ("True", "True", "False", True, True, False),
            ("True", "False", "True", True, False, True),
            ("True", "False", "False", True, False, False),
            ("False", "True", "True", False, True, True),
            ("False", "True", "False", False, True, False),
            ("False", "False", "True", False, False, True),
            ("False", "False", "False", False, False, False),
            # Numeric bools (1 and 0)
            ("1", "1", "1", True, True, True),
            ("0", "0", "0", False, False, False),
        ],
    )
    def test_launch_argument_matrix(
        self,
        enable_meshcat,
        launch_gui,
        rviz,
        expected_meshcat_bool,
        expect_gui_active,
        expect_rviz_active,
    ):
        """Tests that all permutations of launch arguments evaluate without exceptions and configure nodes correctly."""
        mod = load_launch_module()
        ld = mod.generate_launch_description()
        ctx = setup_context_with_args(
            ld,
            {
                "enable_meshcat": enable_meshcat,
                "launch_gui": launch_gui,
                "rviz": rviz,
            },
        )

        nodes = [e for e in ld.entities if isinstance(e, Node)]
        node_map = {str(n._Node__node_name): n for n in nodes}

        # Node 1 and Node 2 must ALWAYS be present and active
        assert "humanoid_centroidal_mpc_sqp_node" in node_map
        assert "drake_mrt_sim_node" in node_map
        assert node_map["humanoid_centroidal_mpc_sqp_node"].condition is None
        assert node_map["drake_mrt_sim_node"].condition is None

        # Node 3: GUI conditional presence
        gui_node = node_map["base_velocity_controller_gui"]
        gui_active = gui_node.condition.evaluate(ctx)
        assert (
            gui_active == expect_gui_active
        ), f"GUI active mismatch: expected {expect_gui_active}, got {gui_active}"

        # Node 4: RViz conditional presence
        rviz_node = node_map["rviz2"]
        rviz_active = rviz_node.condition.evaluate(ctx)
        assert (
            rviz_active == expect_rviz_active
        ), f"RViz active mismatch: expected {expect_rviz_active}, got {rviz_active}"

        # Verify enable_meshcat parameter coercion in drake_mrt_sim_node
        drake_node = node_map["drake_mrt_sim_node"]
        evaluated_params = evaluate_parameters(ctx, drake_node._Node__parameters)
        meshcat_param = None
        for p in evaluated_params:
            if isinstance(p, dict) and "enable_meshcat" in p:
                meshcat_param = p["enable_meshcat"]

        assert isinstance(
            meshcat_param, bool
        ), f"enable_meshcat must be Python bool, got {type(meshcat_param).__name__}"
        assert (
            meshcat_param is expected_meshcat_bool
        ), f"enable_meshcat mismatch: expected {expected_meshcat_bool}, got {meshcat_param}"

    def test_taskset_cpu_affinity_masks(self):
        """Verifies taskset prefixes, exact bitmask calculation, and non-overlap."""
        mod = load_launch_module()
        ld = mod.generate_launch_description()
        ctx = setup_context_with_args(ld, {})

        nodes = {str(n._Node__node_name): n for n in ld.entities if isinstance(n, Node)}

        # Node 1: MPC SQP Node
        mpc_node = nodes["humanoid_centroidal_mpc_sqp_node"]
        mpc_prefix = mpc_node.process_description.prefix
        assert mpc_prefix is not None, "MPC node must have a prefix configured"
        mpc_prefix_str = "".join([ctx.perform_substitution(s) for s in mpc_prefix])
        assert "taskset -c 0-4" in mpc_prefix_str, f"Unexpected MPC prefix: {mpc_prefix_str}"

        mpc_cores = parse_core_range("0-4")
        mpc_mask = cores_to_mask(mpc_cores)
        assert mpc_mask == 0x1F, f"MPC mask must be 0x1f (31), got {hex(mpc_mask)}"

        # Node 2: Drake MRT Sim Node
        drake_node = nodes["drake_mrt_sim_node"]
        drake_prefix = drake_node.process_description.prefix
        assert drake_prefix is not None, "Drake node must have a prefix configured"
        drake_prefix_str = "".join([ctx.perform_substitution(s) for s in drake_prefix])
        assert "taskset -c 5-9" in drake_prefix_str, f"Unexpected Drake prefix: {drake_prefix_str}"

        drake_cores = parse_core_range("5-9")
        drake_mask = cores_to_mask(drake_cores)
        assert drake_mask == 0x3E0, f"Drake mask must be 0x3e0 (992), got {hex(drake_mask)}"

        # Core Isolation & Non-Overlap
        assert (
            mpc_mask & drake_mask
        ) == 0, f"Collision between MPC mask {hex(mpc_mask)} and Drake mask {hex(drake_mask)}"

        # Host 12-core system allocation check
        allocated_mask = mpc_mask | drake_mask
        assert allocated_mask == 0x3FF, f"Allocated mask must be 0x3ff (cores 0-9), got {hex(allocated_mask)}"
        reserved_gui_os_mask = 0xFFF ^ allocated_mask
        assert (
            reserved_gui_os_mask == 0xC00
        ), f"Reserved GUI/OS mask must be 0xc00 (cores 10-11), got {hex(reserved_gui_os_mask)}"

        # Node 3: GUI Node must NOT have a taskset prefix
        gui_node = nodes["base_velocity_controller_gui"]
        gui_prefix = gui_node.process_description.prefix
        gui_prefix_str = "".join([ctx.perform_substitution(s) for s in gui_prefix]) if gui_prefix else ""
        assert "taskset" not in gui_prefix_str, f"GUI node must NOT have taskset prefix: {gui_prefix_str}"

        # Node 4: RViz Node must NOT have a taskset prefix
        rviz_node = nodes["rviz2"]
        rviz_prefix = rviz_node.process_description.prefix
        rviz_prefix_str = "".join([ctx.perform_substitution(s) for s in rviz_prefix]) if rviz_prefix else ""
        assert "taskset" not in rviz_prefix_str, f"RViz node must NOT have taskset prefix: {rviz_prefix_str}"

    def test_node_command_tokenization(self):
        """Verifies that Executable.prepare generates proper CLI tokens with taskset prepended."""
        mod = load_launch_module()
        ld = mod.generate_launch_description()
        ctx = setup_context_with_args(ld, {"launch_gui": "true", "rviz": "true"})

        nodes = [e for e in ld.entities if isinstance(e, Node)]
        for node in nodes:
            node._perform_substitutions(ctx)
            ros_specific = {
                "name": f"__node:={node._Node__expanded_node_name}",
                "ns": f"__ns:={node._Node__expanded_node_namespace}",
            }
            ctx.extend_locals({"ros_specific_arguments": ros_specific})
            node.process_description.prepare(ctx, node)

        node_map = {str(n._Node__node_name): n for n in nodes}

        # MPC SQP node CLI tokens
        mpc_cmd = node_map["humanoid_centroidal_mpc_sqp_node"].process_description.final_cmd
        assert mpc_cmd[:3] == ["taskset", "-c", "0-4"], f"MPC cmd tokens unexpected: {mpc_cmd[:5]}"

        # Drake MRT sim node CLI tokens
        drake_cmd = node_map["drake_mrt_sim_node"].process_description.final_cmd
        assert drake_cmd[:3] == ["taskset", "-c", "5-9"], f"Drake cmd tokens unexpected: {drake_cmd[:5]}"

        # GUI node CLI tokens
        gui_cmd = node_map["base_velocity_controller_gui"].process_description.final_cmd
        assert gui_cmd[0] != "taskset", f"GUI node cmd must not start with taskset: {gui_cmd[:3]}"
        assert "base_velocity_controller_gui" in gui_cmd[0]

        # RViz node CLI tokens
        rviz_cmd = node_map["rviz2"].process_description.final_cmd
        assert rviz_cmd[0] != "taskset", f"RViz node cmd must not start with taskset: {rviz_cmd[:3]}"
        assert "rviz2" in rviz_cmd[0]

    @pytest.mark.parametrize("invalid_val", ["invalid", "2", "none", ""])
    def test_adversarial_invalid_meshcat_argument(self, invalid_val):
        """Verifies that invalid values for enable_meshcat raise ValueError immediately."""
        mod = load_launch_module()
        ld = mod.generate_launch_description()
        ctx = setup_context_with_args(ld, {"enable_meshcat": invalid_val})

        nodes = {str(n._Node__node_name): n for n in ld.entities if isinstance(n, Node)}
        drake_node = nodes["drake_mrt_sim_node"]
        with pytest.raises(ValueError):
            evaluate_parameters(ctx, drake_node._Node__parameters)

    @pytest.mark.parametrize("invalid_val", ["invalid", "2", "none", ""])
    def test_adversarial_invalid_gui_argument(self, invalid_val):
        """Verifies that invalid values for launch_gui raise InvalidConditionExpressionError."""
        mod = load_launch_module()
        ld = mod.generate_launch_description()
        ctx = setup_context_with_args(ld, {"launch_gui": invalid_val})

        nodes = {str(n._Node__node_name): n for n in ld.entities if isinstance(n, Node)}
        gui_node = nodes["base_velocity_controller_gui"]
        with pytest.raises(InvalidConditionExpressionError):
            gui_node.condition.evaluate(ctx)

    def test_drake_sim_parameters_integrity(self):
        """Verifies that all physical, simulation, and contact parameters are accurately set."""
        mod = load_launch_module()
        ld = mod.generate_launch_description()
        ctx = setup_context_with_args(ld, {})

        nodes = {str(n._Node__node_name): n for n in ld.entities if isinstance(n, Node)}
        drake_node = nodes["drake_mrt_sim_node"]
        params = evaluate_parameters(ctx, drake_node._Node__parameters)
        param_dict = {}
        for p in params:
            if isinstance(p, dict):
                param_dict.update(p)

        expected_params = {
            "sim_dt": 0.0005,
            "control_rate_hz": 500.0,
            "joint_damping": 10.0,
            "ground_static_friction": 2.0,
            "ground_dynamic_friction": 1.5,
            "ground_dissipation": 2.0,
            "ground_point_stiffness": 50000.0,
            "force_mujoco_contact_flags": True,
            "reset_on_fall": True,
            "fall_z_threshold": 0.2,
            "reset_cooldown_s": 1.0,
            "enable_meshcat": True,
            "meshcat_host": "*",
            "meshcat_publish_rate_hz": 30.0,
            "verbose": True,
        }

        for key, val in expected_params.items():
            assert key in param_dict, f"Parameter '{key}' missing from drake_mrt_sim_node"
            assert (
                param_dict[key] == val
            ), f"Parameter '{key}' value mismatch: expected {val}, got {param_dict[key]}"
            assert isinstance(
                param_dict[key], type(val)
            ), f"Parameter '{key}' type mismatch: expected {type(val)}, got {type(param_dict[key])}"
