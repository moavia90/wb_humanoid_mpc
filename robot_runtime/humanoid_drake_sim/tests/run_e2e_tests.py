#!/usr/bin/env python3
"""
E2E Test Runner for Unitree G1 2-Node Drake Simulation Architecture.

Supports:
  --tier 1     : Feature Coverage (F1 to F8)
  --tier 2     : Boundary & Corner Cases (B1 to B10)
  --tier 3     : Cross-Feature Combinations (C1 to C5)
  --tier 4     : Real-World Scenarios (S1 to S5)
  --all        : Complete 4-Tier Test Suite
  --dry-run    : Display test suite execution plan without running
  -v, --verbose: Verbose test output
  -s           : Uncaptured test stdout/stderr

Enforces:
  1. Single Simulation Instance Rule via kernel flock on /tmp/humanoid_drake_sim_instance.lock.
  2. Pre- and post-test process cleanup (_kill_sim_nodes).
  3. Port 7000 release verification.
  4. Working directory alignment to workspace root.
"""

import argparse
import fcntl
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Dict, List, Optional


LOCK_FILE = "/tmp/humanoid_drake_sim_instance.lock"
WORKSPACE_ROOT = "/home/mov/Documents/humanoid_mpc_ws"
TESTS_DIR = os.path.join(
    WORKSPACE_ROOT, "src/wb_humanoid_mpc/robot_runtime/humanoid_drake_sim/tests"
)

# Tier Definitions and Test Mapping
TIER_MAPPING: Dict[str, List[Dict[str, str]]] = {
    "1": [
        {
            "file": "test_static_features_and_scope.py",
            "desc": "F1 Documentation & F3 Package Scope Integrity (10 tests)",
        },
        {
            "file": "test_launch_m1.py",
            "desc": "F2 CPU Core Allocation & Launch Configuration (30 tests)",
        },
        {
            "file": "test_ros_interfaces_and_warmup.py",
            "desc": "F4 ROS 2 Topics/Services & F5 Observation Warmup (1 test)",
        },
        {
            "file": "test_meshcat_visualizer.py",
            "desc": "F8 Meshcat 3D Visualizer Integration (1 test)",
        },
        {
            "file": "test_standing_stability.py",
            "desc": "F6 Stable Standing 15-Second Hold (1 test)",
        },
        {
            "file": "test_walking_forward.py",
            "desc": "F7 Forward Walking Locomotion (1 test)",
        },
        {
            "file": "test_walking_backward.py",
            "desc": "F7 Backward Walking Locomotion (1 test)",
        },
    ],
    "2": [
        {
            "file": "test_launch_m1.py",
            "desc": "Parameter Coercion & Adversarial Arguments (30 tests)",
        },
        {
            "file": "test_boundary_corner_cases.py",
            "desc": "B1-B5 Posture Bounds, Steady Drift & B8 Rapid Reversal (2 tests)",
        },
        {
            "file": "test_ros_interfaces_and_warmup.py",
            "desc": "B9 Rapid Reset Service Invocations Under Stance (1 test)",
        },
    ],
    "3": [
        {
            "file": "test_cross_feature_combinations.py",
            "desc": "C1 Meshcat Standing, C2 Meshcat Walking, C4 Walking Reset, C5 Live Affinity (2 tests)",
        },
    ],
    "4": [
        {
            "file": "test_standing_stability.py",
            "desc": "S1 Continuous 15.0s Autonomous Stance (1 test)",
        },
        {
            "file": "test_walking_forward.py",
            "desc": "S2 10.0s Continuous Forward Locomotion (1 test)",
        },
        {
            "file": "test_walking_backward.py",
            "desc": "S3 10.0s Continuous Backward Locomotion (1 test)",
        },
        {
            "file": "test_real_world_scenarios.py",
            "desc": "S4 Multi-Stage Locomotion Mission & S5 Full Production Launch (2 tests)",
        },
    ],
}


def _kill_sim_nodes() -> None:
    """Terminate all lingering simulation nodes, launch processes, and GUIs."""
    targets = [
        ["pkill", "-9", "-x", "drake_mrt_sim_node"],
        ["pkill", "-9", "-x", "humanoid_centroidal_mpc_sqp_node"],
        ["pkill", "-9", "-f", "drake_mrt_sim_node"],
        ["pkill", "-9", "-f", "humanoid_centroidal_mpc_sqp_node"],
        ["pkill", "-9", "-f", "drake_mrt_sim.launch.py"],
        ["pkill", "-9", "-f", "walking_velocity_command"],
        ["pkill", "-9", "-f", "base_velocity_controller_gui"],
    ]
    for cmd in targets:
        try:
            subprocess.run(cmd, capture_output=True, timeout=5.0)
        except Exception:
            pass


def _wait_port_freed(port: int = 7000, timeout: float = 5.0) -> None:
    """Ensure socket port is released before starting the next test."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(0.2)


def get_unique_test_plan(tier: str) -> List[Dict[str, str]]:
    """Resolve ordered list of tests for the requested tier or all tiers."""
    if tier.lower() in ("all", "1-4"):
        # Distinct sequential order covering all tiers
        all_ordered_files = [
            ("test_static_features_and_scope.py", "Tier 1: F1 Documentation & F3 Scope Boundary"),
            ("test_launch_m1.py", "Tier 1 & 2: F2 Launch Core Allocation & Boundary Arguments"),
            ("test_ros_interfaces_and_warmup.py", "Tier 1 & 2: F4 Topics/Services, F5 Warmup & B9 Reset"),
            ("test_meshcat_visualizer.py", "Tier 1: F8 Meshcat 3D Visualizer & Port 7000"),
            ("test_standing_stability.py", "Tier 1 & 4: F6 Stable Standing & S1 15s Autonomous Stance"),
            ("test_walking_forward.py", "Tier 1 & 4: F7 Forward Walking & S2 10s Forward Locomotion"),
            ("test_walking_backward.py", "Tier 1 & 4: F7 Backward Walking & S3 10s Backward Locomotion"),
            ("test_boundary_corner_cases.py", "Tier 2: B1-B5 Posture Bounds, Steady Drift & B8 Rapid Reversal"),
            ("test_cross_feature_combinations.py", "Tier 3: C1 Meshcat Stance, C2 Meshcat Walk, C4 Reset, C5 Affinity"),
            ("test_real_world_scenarios.py", "Tier 4: S4 Multi-Stage Mission & S5 Production Launch"),
        ]
        return [{"file": f, "desc": d} for f, d in all_ordered_files]

    if tier in TIER_MAPPING:
        return TIER_MAPPING[tier]

    raise ValueError(f"Unknown tier: {tier}. Expected '1', '2', '3', '4', or 'all'.")


def run_e2e_suite(
    tier: str,
    verbose: bool = False,
    stream_output: bool = True,
    dry_run: bool = False,
) -> int:
    """Universal E2E runner enforcing Single Simulation Instance and reporting results."""
    test_plan = get_unique_test_plan(tier)

    print("=" * 80)
    print(f" UNITREE G1 2-NODE DRAKE SIMULATION E2E TEST RUNNER")
    print(f" Target Suite: Tier {tier.upper()} ({len(test_plan)} test target files)")
    print(f" Working Dir:  {WORKSPACE_ROOT}")
    print(f" Lock Mutex:   {LOCK_FILE}")
    print("=" * 80)

    if dry_run:
        print("\n[DRY RUN] Planned test execution sequence:")
        for idx, item in enumerate(test_plan, 1):
            print(f"  {idx:2d}. {item['file']:<35} - {item['desc']}")
        print("\nDry run completed successfully. Zero tests executed.")
        return 0

    # Verify exclusive lock availability
    try:
        lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    except (BlockingIOError, OSError) as exc:
        print(f"\n[FATAL] Concurrency violation: {LOCK_FILE} is held by another process: {exc}")
        print("The Single Simulation Instance Rule prohibits simultaneous simulation executions.")
        return 1

    results = []
    total_start = time.time()

    try:
        # Pre-execution environment cleanup
        _kill_sim_nodes()
        _wait_port_freed(7000, timeout=3.0)

        # Prepare execution environment
        run_env = os.environ.copy()
        ws_setup = os.path.join(WORKSPACE_ROOT, "install/setup.bash")
        if os.path.exists(ws_setup) and "/home/mov/Documents/humanoid_mpc_ws/install" not in run_env.get("AMENT_PREFIX_PATH", ""):
            out = subprocess.check_output(f"source {ws_setup} && env", shell=True, executable="/bin/bash", text=True)
            for line in out.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    run_env[k] = v

        for idx, item in enumerate(test_plan, 1):
            target_file = item["file"]
            desc = item["desc"]
            file_path = os.path.join(TESTS_DIR, target_file)

            print(f"\n[{idx}/{len(test_plan)}] Running {target_file} ({desc})...")
            test_start = time.time()

            # Ensure previous run leaves clean state
            _kill_sim_nodes()
            _wait_port_freed(7000, timeout=2.0)

            pytest_cmd = [
                sys.executable,
                "-m",
                "pytest",
                "-o", "filterwarnings=ignore::pytest.PytestUnknownMarkWarning",
                file_path,
            ]
            if verbose:
                pytest_cmd.append("-v")
            if stream_output:
                pytest_cmd.append("-s")

            proc = subprocess.run(
                pytest_cmd,
                cwd=WORKSPACE_ROOT,
                env=run_env,
                capture_output=not stream_output,
                text=True,
            )

            duration = time.time() - test_start
            passed = (proc.returncode == 0)
            status_str = "PASS" if passed else "FAIL"

            results.append({
                "file": target_file,
                "desc": desc,
                "status": status_str,
                "duration": duration,
                "code": proc.returncode,
            })

            print(f"[{status_str}] {target_file} completed in {duration:.2f}s (code {proc.returncode})")

            # Post-test cleanup
            _kill_sim_nodes()
            _wait_port_freed(7000, timeout=3.0)

    finally:
        _kill_sim_nodes()

    total_duration = time.time() - total_start

    # Print Formatted ASCII Summary Table
    print("\n" + "=" * 90)
    print(f"                               E2E TEST SUMMARY REPORT")
    print("=" * 90)
    print(f"{'#':<3} | {'Test Target File':<35} | {'Status':<6} | {'Duration':<8} | {'Description'}")
    print("-" * 90)

    pass_count = 0
    fail_count = 0

    for idx, r in enumerate(results, 1):
        if r["status"] == "PASS":
            pass_count += 1
        else:
            fail_count += 1
        print(f"{idx:<3} | {r['file']:<35} | {r['status']:<6} | {r['duration']:>6.2f}s | {r['desc']}")

    print("-" * 90)
    print(f"Total Suites: {len(results)} | Passed: {pass_count} | Failed: {fail_count} | Total Wall Time: {total_duration:.2f}s")
    print("=" * 90)

    if fail_count == 0:
        print(f"\n>>> [SUCCESS] 100% of E2E tests PASSED! Certification complete for Tier {tier.upper()}.\n")
        return 0
    else:
        print(f"\n>>> [FAILURE] {fail_count} test suite(s) failed. See logs above for details.\n")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal CLI E2E Test Suite Runner for Unitree G1 2-Node Drake Simulation."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tier",
        type=str,
        choices=["1", "2", "3", "4", "all"],
        help="Specify test tier to execute (1, 2, 3, 4, or all).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Alias for --tier all (executes complete 4-tier test suite).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output in underlying pytest runs.",
    )
    parser.add_argument(
        "-s", "--nocapture",
        action="store_true",
        default=True,
        help="Disable stdout/stderr capture for real-time streaming (default: True).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display scheduled test cases without executing simulations.",
    )

    args = parser.parse_args()
    tier_choice = "all" if args.all else args.tier

    exit_code = run_e2e_suite(
        tier=tier_choice,
        verbose=args.verbose,
        stream_output=args.nocapture,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
