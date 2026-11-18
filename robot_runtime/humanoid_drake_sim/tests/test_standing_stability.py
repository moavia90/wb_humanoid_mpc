#!/usr/bin/env python3
"""
Automated Headless Standing Stability Verification Test Suite (Requirement R4.1).

Verifies:
1. Continuous standing for >= 15.0 s simulated time.
2. Pelvis height z in [0.65, 0.90] m across all samples.
3. Pelvis roll and pitch in [-0.35, +0.35] rad across all samples.
4. falls == 0 across the entire duration.
5. Single Simulation Instance enforcement via fcntl file locking and pkill cleanup.
6. ROS_DOMAIN_ID isolation.
"""

import fcntl
import os
import re
import signal
import subprocess
import time
import pytest

TELEMETRY_REGEX = re.compile(
    r"\[standing_telemetry\]\s+t_sim=([0-9.]+)\s+s\s+\|\s+pelvis_z=([0-9.]+)\s+m\s+\|\s+roll=([0-9.-]+)\s+rad\s+\|\s+pitch=([0-9.-]+)\s+rad\s+\|\s+falls=(\d+)"
)


class TestStandingStabilityR41:
    """Automated E2E Headless Test for Requirement R4.1."""

    LOCK_FILE = "/tmp/humanoid_drake_sim_instance.lock"

    @pytest.fixture(autouse=True)
    def single_simulation_guard_and_cleanup(self):
        """
        Enforce the Single Simulation Instance rule:
        1. Acquire exclusive non-blocking file lock to prevent concurrent test runs.
        2. Clean any orphaned simulation processes before test starts.
        3. Clean all simulation processes after test completes.
        """
        lock_fd = os.open(self.LOCK_FILE, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(lock_fd)
            pytest.fail(
                f"CONCURRENCY VIOLATION: Simulation lock ({self.LOCK_FILE}) is already held. "
                f"Single Simulation Instance Rule prohibits simultaneous execution! {exc}"
            )

        self._kill_sim_nodes()
        time.sleep(1.0)

        yield

        self._kill_sim_nodes()
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        except Exception:
            pass

    @staticmethod
    def _kill_sim_nodes():
        """Forcefully terminate simulation nodes and launch wrappers."""
        for target in [
            ["pkill", "-9", "-x", "drake_mrt_sim_node"],
            ["pkill", "-9", "-x", "humanoid_centroidal_mpc_sqp_node"],
            ["pkill", "-9", "-f", "drake_mrt_sim_node"],
            ["pkill", "-9", "-f", "humanoid_centroidal_mpc_sqp_node"],
            ["pkill", "-9", "-f", "drake_mrt_sim.launch.py"],
            ["pkill", "-9", "-f", "walking_velocity_command"],
        ]:
            try:
                subprocess.run(target, capture_output=True, timeout=5.0)
            except Exception:
                pass

    def test_standing_stability_15_seconds(self):
        """Launches drake_mrt_sim.launch.py headless and verifies 15s stable standing."""
        test_env = os.environ.copy()
        test_env["ROS_DOMAIN_ID"] = str((os.getpid() % 100) + 11)

        ws_setup = "/home/mov/Documents/humanoid_mpc_ws/install/setup.bash"
        if os.path.exists(ws_setup) and "/home/mov/Documents/humanoid_mpc_ws/install" not in test_env.get("AMENT_PREFIX_PATH", ""):
            out = subprocess.check_output(f"source {ws_setup} && env", shell=True, executable="/bin/bash", text=True)
            for line in out.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    test_env[k] = v

        cmd = [
            "ros2", "launch", "g1_centroidal_mpc", "drake_mrt_sim.launch.py",
            "enable_meshcat:=false",
            "launch_gui:=false",
            "rviz:=false",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
            env=test_env,
            cwd="/home/mov/Documents/humanoid_mpc_ws",
        )

        telemetry_samples = []
        max_sim_time = 0.0
        start_wall_time = time.time()
        timeout_wall_time = 90.0  # Allow 90s wall time for 15s sim time

        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break

                match = TELEMETRY_REGEX.search(line)
                if match:
                    t_sim = float(match.group(1))
                    pelvis_z = float(match.group(2))
                    roll = float(match.group(3))
                    pitch = float(match.group(4))
                    falls = int(match.group(5))

                    telemetry_samples.append({
                        "t_sim": t_sim,
                        "pelvis_z": pelvis_z,
                        "roll": roll,
                        "pitch": pitch,
                        "falls": falls,
                    })

                    max_sim_time = max(max_sim_time, t_sim)

                    # Continuous Invariant Assertions
                    assert falls == 0, f"Fall detected at t_sim={t_sim} s (falls={falls})"
                    assert 0.65 <= pelvis_z <= 0.90, (
                        f"Pelvis height out of bounds at t_sim={t_sim} s: {pelvis_z:.4f} m "
                        f"(expected [0.65, 0.90] m)"
                    )
                    assert abs(roll) <= 0.35, (
                        f"Pelvis roll exceeded limit at t_sim={t_sim} s: {roll:.4f} rad "
                        f"(expected |roll| <= 0.35 rad)"
                    )
                    assert abs(pitch) <= 0.35, (
                        f"Pelvis pitch exceeded limit at t_sim={t_sim} s: {pitch:.4f} rad "
                        f"(expected |pitch| <= 0.35 rad)"
                    )

                    if max_sim_time >= 15.0:
                        break

                if time.time() - start_wall_time > timeout_wall_time:
                    pytest.fail(
                        f"Test timed out after {timeout_wall_time}s wall time. "
                        f"Achieved t_sim={max_sim_time:.3f}s (required >= 15.0s)"
                    )

        finally:
            # Graceful process group termination
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self._kill_sim_nodes()

        # Terminal Aggregate Assertions
        assert len(telemetry_samples) >= 25, (
            f"Insufficient telemetry samples received: {len(telemetry_samples)} (expected >= 25)"
        )
        assert max_sim_time >= 15.0, (
            f"Maximum simulated time {max_sim_time:.3f} s did not reach 15.0 s requirement"
        )
        print(f"\n[Test Result] Standing stability verified! Collected {len(telemetry_samples)} samples, max t_sim={max_sim_time:.3f}s")
        print(f"First telemetry sample:  {telemetry_samples[0]}")
        print(f"Middle telemetry sample: {telemetry_samples[len(telemetry_samples)//2]}")
        print(f"Final telemetry sample:  {telemetry_samples[-1]}")
