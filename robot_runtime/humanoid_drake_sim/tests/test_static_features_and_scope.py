#!/usr/bin/env python3
"""
Tier 1: Static Documentation and Scope Verification Test Suite (F1 & F3).

Verifies:
- F1.1–F1.5: Documentation.md completeness, 2-node architecture, taskset core allocation,
             MuJoCo failure analysis, and startup warmup protocol.
- F3.1–F3.5: Requirement R3 scope compliance via git diff (zero modifications to protected packages).
"""

import os
import subprocess
import pytest


DOC_PATH = "/home/mov/Documents/humanoid_mpc_ws/src/wb_humanoid_mpc/robot_runtime/humanoid_drake_sim/Documentation.md"
REPO_ROOT = "/home/mov/Documents/humanoid_mpc_ws/src/wb_humanoid_mpc"


@pytest.fixture(scope="module")
def doc_text():
    """Load Documentation.md content once for all documentation checks."""
    assert os.path.isfile(DOC_PATH), f"Documentation file missing at {DOC_PATH}"
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    return content


@pytest.mark.tier1
class TestDocumentationCompletenessF1:
    """Feature 1: Documentation verification (F1.1 - F1.5)."""

    def test_f1_1_documentation_exists_and_substantial(self, doc_text):
        """F1.1: Verify Documentation.md exists and is substantial (> 10,000 bytes)."""
        file_size = len(doc_text.encode("utf-8"))
        assert file_size >= 10000, f"Documentation.md is smaller than expected: {file_size} bytes"

    def test_f1_2_two_node_architecture_documented(self, doc_text):
        """F1.2: Verify Documentation.md contains 2-Node ROS 2 architecture details."""
        required_tokens = [
            "humanoid_centroidal_mpc_sqp_node",
            "drake_mrt_sim_node",
            "/g1/mpc_policy",
            "/g1/mpc_observation",
            "/g1/mpc_reset",
        ]
        for token in required_tokens:
            assert token in doc_text, f"Documentation.md missing required architectural token: '{token}'"

    def test_f1_3_taskset_core_allocation_documented(self, doc_text):
        """F1.3: Verify Documentation.md documents taskset CPU core allocations (0-4, 5-9, 10-11)."""
        required_allocations = [
            "0-4",
            "5-9",
            "10-11",
            "taskset",
        ]
        for alloc in required_allocations:
            assert alloc in doc_text, f"Documentation.md missing taskset allocation specification: '{alloc}'"

    def test_f1_4_mujoco_failure_reasons_analyzed(self, doc_text):
        """F1.4: Verify Documentation.md explains single-process MuJoCo failure reason in Drake."""
        required_concepts = [
            "OpenMP",
            "SAP",
            "jitter",
            "CentroidalMpcRobotSim",
        ]
        for concept in required_concepts:
            assert concept.lower() in doc_text.lower(), (
                f"Documentation.md missing root-cause failure analysis concept: '{concept}'"
            )

    def test_f1_5_startup_warmup_protocol_documented(self, doc_text):
        """F1.5: Verify Documentation.md documents startup/reset warm-up synchronization protocol."""
        required_steps = [
            "TargetTrajectoriesCalculatorBase",
            "warm-up",
            "0.7925",
        ]
        for step in required_steps:
            assert step.lower() in doc_text.lower(), (
                f"Documentation.md missing startup warmup protocol detail: '{step}'"
            )


@pytest.mark.tier1
class TestPackageScopeComplianceF3:
    """Feature 3: Strict package scope compliance verification (F3.1 - F3.5)."""

    def _get_git_diff(self, relative_path: str) -> str:
        """Run git diff against HEAD for a specific subpath."""
        res = subprocess.run(
            ["git", "diff", "HEAD", "--", relative_path],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()

    def test_f3_1_scope_centroidal_mpc(self):
        """F3.1: Confirm zero changes to humanoid_centroidal_mpc packages."""
        diff_core = self._get_git_diff("humanoid_nmpc/humanoid_centroidal_mpc")
        diff_ros2 = self._get_git_diff("humanoid_nmpc/humanoid_centroidal_mpc_ros2")
        assert diff_core == "", f"Protected package humanoid_centroidal_mpc has uncommitted changes:\n{diff_core}"
        assert diff_ros2 == "", f"Protected package humanoid_centroidal_mpc_ros2 has uncommitted changes:\n{diff_ros2}"

    def test_f3_2_scope_common_mpc(self):
        """F3.2: Confirm zero changes to humanoid_common_mpc packages."""
        diff_common = self._get_git_diff("humanoid_nmpc/humanoid_common_mpc")
        diff_common_ros2 = self._get_git_diff("humanoid_nmpc/humanoid_common_mpc_ros2")
        assert diff_common == "", f"Protected package humanoid_common_mpc has uncommitted changes:\n{diff_common}"
        assert diff_common_ros2 == "", f"Protected package humanoid_common_mpc_ros2 has uncommitted changes:\n{diff_common_ros2}"

    def test_f3_3_scope_wb_mpc(self):
        """F3.3: Confirm zero changes to humanoid_wb_mpc."""
        diff_wb = self._get_git_diff("humanoid_nmpc/humanoid_wb_mpc")
        assert diff_wb == "", f"Protected package humanoid_wb_mpc has uncommitted changes:\n{diff_wb}"

    def test_f3_4_scope_mujoco_sim_interface(self):
        """F3.4: Confirm zero changes to mujoco_sim_interface."""
        diff_mujoco = self._get_git_diff("robot_runtime/mujoco_sim_interface")
        assert diff_mujoco == "", f"Protected package mujoco_sim_interface has uncommitted changes:\n{diff_mujoco}"

    def test_f3_5_scope_remote_control(self):
        """F3.5: Confirm zero changes to remote_control."""
        diff_rc = self._get_git_diff("humanoid_nmpc/remote_control")
        assert diff_rc == "", f"Protected package remote_control has uncommitted changes:\n{diff_rc}"
