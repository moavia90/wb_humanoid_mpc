#!/usr/bin/env python3
"""
Tier 2: Boundary & Corner Cases Test Suite (B1-B5, B8).

Verifies:
- B1 & B2: Pelvis height bounds under continuous control (z in [0.65, 0.90] m).
- B3 & B4: Roll and pitch posture bounds (|roll| <= 0.35 rad, |pitch| <= 0.35 rad).
- B5: Zero-velocity command steady-state drift (|Delta x| < 0.05 m over 5 s).
- B8: Rapid directional velocity command reversal (+0.40 -> -0.40 m/s) with zero falls.
"""

import fcntl
import os
import re
import signal
import subprocess
import time
import pytest


LOCK_FILE = "/tmp/humanoid_drake_sim_instance.lock"

STANDING_TELEMETRY_REGEX = re.compile(
    r"\[standing_telemetry\]\s+t_sim=([0-9.]+)\s+s\s+\|\s+pelvis_z=([0-9.]+)\s+m\s+\|\s+roll=([0-9.-]+)\s+rad\s+\|\s+pitch=([0-9.-]+)\s+rad\s+\|\s+falls=(\d+)"
)
WALKING_TELEMETRY_REGEX = re.compile(
    r"\[walking_telemetry\]\s+t_sim=(?P<t_sim>[0-9.]+)\s+s\s+\|\s+pelvis_x=(?P<pelvis_x>[0-9.-]+)\s+m\s+\|\s+pelvis_z=(?P<pelvis_z>[0-9.]+)\s+m\s+\|\s+falls=(?P<falls>\d+)"
)


@pytest.mark.tier2
class TestBoundaryCornerCases:
    """Automated Tier 2 Boundary & Corner Case Tests."""

    @classmethod
    def _kill_sim_nodes(cls):
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

    @pytest.fixture(autouse=True)
    def single_simulation_guard_and_cleanup(self):
        """Enforce Single Simulation Instance rule via kernel flock."""
        lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(lock_fd)
            pytest.fail(f"CONCURRENCY VIOLATION: Simulation lock held: {exc}")

        self._kill_sim_nodes()
        time.sleep(1.0)
        yield
        self._kill_sim_nodes()
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        except Exception:
            pass

    def _setup_env(self, domain_offset=15):
        """Configure isolated environment with workspace paths."""
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str((os.getpid() % 50) + domain_offset)
        ws_setup = "/home/mov/Documents/humanoid_mpc_ws/install/setup.bash"
        if os.path.exists(ws_setup) and "/home/mov/Documents/humanoid_mpc_ws/install" not in env.get("AMENT_PREFIX_PATH", ""):
            out = subprocess.check_output(f"source {ws_setup} && env", shell=True, executable="/bin/bash", text=True)
            for line in out.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v
        return env

    def test_steady_state_drift_and_posture_boundaries(self):
        """B1-B5: Verifies posture bounds and zero-velocity drift over 7s standing."""
        test_env = self._setup_env(domain_offset=15)
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
        walking_samples = []
        max_sim_time = 0.0
        start_wall_time = time.time()
        timeout_wall_time = 45.0

        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break

                std_match = STANDING_TELEMETRY_REGEX.search(line)
                if std_match:
                    t_sim = float(std_match.group(1))
                    pelvis_z = float(std_match.group(2))
                    roll = float(std_match.group(3))
                    pitch = float(std_match.group(4))
                    falls = int(std_match.group(5))

                    telemetry_samples.append({
                        "t_sim": t_sim, "pelvis_z": pelvis_z,
                        "roll": roll, "pitch": pitch, "falls": falls
                    })
                    max_sim_time = max(max_sim_time, t_sim)

                    # B1 & B2: Height bounds
                    assert 0.65 <= pelvis_z <= 0.90, f"Height bound violated at t_sim={t_sim}s: {pelvis_z}m"
                    # B3 & B4: Roll / pitch limits
                    assert abs(roll) <= 0.35, f"Roll bound violated at t_sim={t_sim}s: {roll} rad"
                    assert abs(pitch) <= 0.35, f"Pitch bound violated at t_sim={t_sim}s: {pitch} rad"
                    assert falls == 0, f"Fall detected at t_sim={t_sim}s"

                    if max_sim_time >= 7.0:
                        break

                wlk_match = WALKING_TELEMETRY_REGEX.search(line)
                if wlk_match:
                    walking_samples.append({
                        "t_sim": float(wlk_match.group("t_sim")),
                        "pelvis_x": float(wlk_match.group("pelvis_x")),
                    })

                if time.time() - start_wall_time > timeout_wall_time:
                    pytest.fail(f"Test timed out waiting for 7.0s sim time (reached {max_sim_time:.2f}s)")

        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self._kill_sim_nodes()

        assert len(telemetry_samples) >= 12, f"Insufficient samples: {len(telemetry_samples)}"
        # B5: Zero-velocity drift test
        # Compare pelvis_x at t_sim ~ 1.5s vs t_sim ~ 6.5s
        x_early = [s["pelvis_x"] for s in walking_samples if 1.0 <= s["t_sim"] <= 2.0]
        x_late = [s["pelvis_x"] for s in walking_samples if 5.5 <= s["t_sim"] <= 7.0]
        assert len(x_early) > 0 and len(x_late) > 0, "Missing walking telemetry samples for drift evaluation"
        drift = abs(x_late[-1] - x_early[0])
        print(f"\n[B5 Result] 5-second horizontal drift: {drift:.4f} m (Requirement: < 0.05 m)")
        assert drift < 0.05, f"Zero-velocity drift excessive: {drift:.4f} m >= 0.05 m"

    def test_rapid_velocity_command_reversal(self):
        """B8: Verifies smooth transition from forward to backward walking (+0.40 -> -0.40 m/s)."""
        test_env = self._setup_env(domain_offset=18)
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

        pub_proc = None
        command_forward = False
        command_backward = False
        forward_start_x = None
        peak_x = None
        final_x = None
        start_wall_time = time.time()
        timeout_wall_time = 75.0

        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break

                # Posture stability checks
                std_match = STANDING_TELEMETRY_REGEX.search(line)
                if std_match:
                    t_sim = float(std_match.group(1))
                    pelvis_z = float(std_match.group(2))
                    roll = float(std_match.group(3))
                    pitch = float(std_match.group(4))
                    falls = int(std_match.group(5))
                    assert falls == 0, f"Fall detected at t_sim={t_sim}s"
                    assert 0.65 <= pelvis_z <= 0.90, f"Height violation at t_sim={t_sim}s: {pelvis_z}m"
                    assert abs(roll) <= 0.35 and abs(pitch) <= 0.35

                wlk_match = WALKING_TELEMETRY_REGEX.search(line)
                if wlk_match:
                    t_sim = float(wlk_match.group("t_sim"))
                    pelvis_x = float(wlk_match.group("pelvis_x"))
                    falls = int(wlk_match.group("falls"))
                    assert falls == 0, f"Fall detected in walking telemetry at t_sim={t_sim}s"

                    # Stage 1: Stabilize until t=1.5s, then command forward
                    if t_sim >= 1.5 and not command_forward:
                        command_forward = True
                        forward_start_x = pelvis_x
                        pub_cmd = [
                            "ros2", "topic", "pub", "--rate", "10",
                            "--qos-reliability", "best_effort",
                            "/humanoid/walking_velocity_command",
                            "humanoid_mpc_msgs/msg/WalkingVelocityCommand",
                            "{linear_velocity_x: 0.4, linear_velocity_y: 0.0, angular_velocity_z: 0.0, desired_pelvis_height: 0.7925}",
                        ]
                        pub_proc = subprocess.Popen(
                            pub_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid, env=test_env
                        )
                        print(f"\n[Reversal Test] Forward command dispatched at t_sim={t_sim:.2f}s (x={pelvis_x:.3f}m)")

                    # Stage 2: Once net forward displacement >= +0.15m, rapidly reverse to -0.40 m/s
                    if command_forward and not command_backward and (pelvis_x - forward_start_x >= 0.15):
                        command_backward = True
                        peak_x = pelvis_x
                        if pub_proc:
                            try:
                                os.killpg(os.getpgid(pub_proc.pid), signal.SIGKILL)
                            except Exception:
                                pass
                        pub_cmd = [
                            "ros2", "topic", "pub", "--rate", "10",
                            "--qos-reliability", "best_effort",
                            "/humanoid/walking_velocity_command",
                            "humanoid_mpc_msgs/msg/WalkingVelocityCommand",
                            "{linear_velocity_x: -0.4, linear_velocity_y: 0.0, angular_velocity_z: 0.0, desired_pelvis_height: 0.7925}",
                        ]
                        pub_proc = subprocess.Popen(
                            pub_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid, env=test_env
                        )
                        print(f"[Reversal Test] RAPID REVERSAL to -0.40 m/s dispatched at x={pelvis_x:.3f}m")

                    # Stage 3: Verify reversal displacement
                    if command_backward:
                        final_x = pelvis_x
                        delta_rev = peak_x - final_x
                        if delta_rev >= 0.15:
                            print(f"[Reversal Test] Reversal displacement achieved: Delta x = -{delta_rev:.3f} m (Peak={peak_x:.3f}m, Final={final_x:.3f}m)")
                            break

                if time.time() - start_wall_time > timeout_wall_time:
                    pytest.fail(f"Reversal test timed out after {timeout_wall_time}s")

        finally:
            if pub_proc:
                try:
                    os.killpg(os.getpgid(pub_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self._kill_sim_nodes()

        assert command_forward, "Forward command was never dispatched"
        assert command_backward, "Backward reversal command was never dispatched"
        assert peak_x is not None and final_x is not None
        reversal_disp = peak_x - final_x
        assert reversal_disp >= 0.10, f"Insufficient backward displacement after reversal: {reversal_disp:.3f}m (< 0.10m)"
        print(f"\n[B8 Result] Reversal verified successfully! Peak X={peak_x:.4f}m, Final X={final_x:.4f}m, Net Return={reversal_disp:.4f}m, Falls=0")
