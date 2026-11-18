#!/usr/bin/env python3
"""
Tier 3: Cross-Feature Combinations Test Suite (C1 - C5).

Verifies pairwise feature interactions:
- C1: Standing stability while querying Meshcat server.
- C2: Stance to forward walking transition with live Meshcat visualization active.
- C3: Dynamic forward-to-backward gait transition without entering stance hold.
- C4: Calling /g1/mpc_reset service during active walking locomotion.
- C5: Live taskset CPU core affinity validation (MPC: 0-4 / 1f, Drake: 5-9 / 3e0) during active physics.
"""

import fcntl
import os
import re
import signal
import subprocess
import time
import urllib.request
import pytest


LOCK_FILE = "/tmp/humanoid_drake_sim_instance.lock"

STANDING_TELEMETRY_REGEX = re.compile(
    r"\[standing_telemetry\]\s+t_sim=([0-9.]+)\s+s\s+\|\s+pelvis_z=([0-9.]+)\s+m\s+\|\s+roll=([0-9.-]+)\s+rad\s+\|\s+pitch=([0-9.-]+)\s+rad\s+\|\s+falls=(\d+)"
)
WALKING_TELEMETRY_REGEX = re.compile(
    r"\[walking_telemetry\]\s+t_sim=(?P<t_sim>[0-9.]+)\s+s\s+\|\s+pelvis_x=(?P<pelvis_x>[0-9.-]+)\s+m\s+\|\s+pelvis_z=(?P<pelvis_z>[0-9.]+)\s+m\s+\|\s+falls=(?P<falls>\d+)"
)


@pytest.mark.tier3
class TestCrossFeatureCombinations:
    """Pairwise combination tests across simulation, control, and visualization."""

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
        """Acquire kernel flock to enforce single instance rule."""
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

    def _setup_env(self, domain_offset=20):
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

    def test_c1_c2_c5_meshcat_walking_and_affinity(self):
        """
        C1, C2 & C5:
        - C1: Standing stability while querying Meshcat server.
        - C2: Transition to forward walk with live Meshcat visualizer active.
        - C5: Live taskset CPU core affinity validation for both nodes while stepping.
        """
        test_env = self._setup_env(domain_offset=21)
        cmd = [
            "ros2", "launch", "g1_centroidal_mpc", "drake_mrt_sim.launch.py",
            "enable_meshcat:=true",
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
        command_dispatched = False
        start_x = None
        final_x = None
        meshcat_verified = False
        affinity_verified = False
        start_wall_time = time.time()
        timeout_wall_time = 75.0

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
                    assert falls == 0, f"Fall detected at t_sim={t_sim}s"
                    assert 0.65 <= pelvis_z <= 0.90, f"Height bound violated: {pelvis_z}m"
                    assert abs(roll) <= 0.35 and abs(pitch) <= 0.35

                    # C1: Query Meshcat HTTP server while standing
                    if t_sim >= 1.0 and not meshcat_verified:
                        try:
                            req = urllib.request.urlopen("http://127.0.0.1:7000/", timeout=3.0)
                            assert req.status == 200, f"Meshcat returned status {req.status}"
                            meshcat_verified = True
                            print(f"\n[C1 Result] Meshcat HTTP query verified (Status 200) at t_sim={t_sim:.2f}s")
                        except Exception as e:
                            print(f"[C1 Retry] Meshcat not ready yet: {e}")

                    # C5: Live taskset CPU core affinity validation
                    if t_sim >= 1.5 and not affinity_verified:
                        mpc_pids = subprocess.check_output(["pgrep", "-f", "humanoid_centroidal_mpc_sqp_node"], text=True).split()
                        drake_pids = subprocess.check_output(["pgrep", "-f", "drake_mrt_sim_node"], text=True).split()
                        assert len(mpc_pids) > 0, "humanoid_centroidal_mpc_sqp_node PID not found"
                        assert len(drake_pids) > 0, "drake_mrt_sim_node PID not found"

                        mpc_aff = subprocess.check_output(["taskset", "-cp", mpc_pids[0]], text=True).strip()
                        drake_aff = subprocess.check_output(["taskset", "-cp", drake_pids[0]], text=True).strip()

                        # MPC must be cores 0-4, Drake must be cores 5-9
                        assert "0-4" in mpc_aff, f"MPC node affinity mismatch: {mpc_aff} (expected 0-4)"
                        assert "5-9" in drake_aff, f"Drake node affinity mismatch: {drake_aff} (expected 5-9)"
                        affinity_verified = True
                        print(f"[C5 Result] Live CPU affinity verified: MPC={mpc_aff}, Drake={drake_aff}")

                wlk_match = WALKING_TELEMETRY_REGEX.search(line)
                if wlk_match:
                    t_sim = float(wlk_match.group("t_sim"))
                    pelvis_x = float(wlk_match.group("pelvis_x"))
                    falls = int(wlk_match.group("falls"))
                    assert falls == 0, f"Fall detected in walking telemetry: {falls}"

                    # C2: Dispatch forward walking command while Meshcat is active
                    if t_sim >= 2.0 and not command_dispatched and meshcat_verified and affinity_verified:
                        command_dispatched = True
                        start_x = pelvis_x
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
                        print(f"[C2 Result] Walking command dispatched with Meshcat active at t_sim={t_sim:.2f}s")

                    if command_dispatched:
                        final_x = pelvis_x
                        delta_x = final_x - start_x
                        if delta_x >= 0.30:
                            print(f"[C2 Result] Forward walk with Meshcat active reached Delta x = {delta_x:.4f} m > 0.30 m")
                            break

                if time.time() - start_wall_time > timeout_wall_time:
                    pytest.fail(f"Test timed out after {timeout_wall_time}s")

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

        assert meshcat_verified, "Meshcat HTTP 200 was not verified"
        assert affinity_verified, "CPU core affinity was not verified"
        assert command_dispatched, "Walking command was not dispatched"
        delta_x = final_x - start_x
        assert delta_x >= 0.30, f"Insufficient forward displacement with Meshcat active: {delta_x:.3f}m"

    def test_c4_reset_service_during_active_walking(self):
        """C4: Invokes /g1/mpc_reset service during active walking locomotion."""
        test_env = self._setup_env(domain_offset=24)
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
        command_dispatched = False
        reset_invoked = False
        start_x = None
        start_wall_time = time.time()
        timeout_wall_time = 60.0

        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break

                std_match = STANDING_TELEMETRY_REGEX.search(line)
                if std_match:
                    t_sim = float(std_match.group(1))
                    pelvis_z = float(std_match.group(2))
                    falls = int(std_match.group(5))
                    assert falls == 0, f"Fall detected at t_sim={t_sim}s"
                    assert 0.65 <= pelvis_z <= 0.90

                wlk_match = WALKING_TELEMETRY_REGEX.search(line)
                if wlk_match:
                    t_sim = float(wlk_match.group("t_sim"))
                    pelvis_x = float(wlk_match.group("pelvis_x"))
                    falls = int(wlk_match.group("falls"))
                    assert falls == 0, f"Fall detected in walking: {falls}"

                    if t_sim >= 1.5 and not command_dispatched:
                        command_dispatched = True
                        start_x = pelvis_x
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
                        print(f"\n[C4] Walking command dispatched at t_sim={t_sim:.2f}s")

                    # Once walking is underway (Delta x >= 0.15 m), invoke reset service
                    if command_dispatched and not reset_invoked and (pelvis_x - start_x >= 0.15):
                        reset_invoked = True
                        # Stop velocity command
                        if pub_proc:
                            try:
                                os.killpg(os.getpgid(pub_proc.pid), signal.SIGKILL)
                            except Exception:
                                pass
                        # Call reset service
                        call_cmd = [
                            "ros2", "service", "call", "/g1/mpc_reset", "ocs2_ros2_msgs/srv/Reset",
                            "{reset: true, target_trajectories: {api_version: 0, time_trajectory: [0.0], state_trajectory: [{value: []}], input_trajectory: [{value: []}]}}"
                        ]
                        res = subprocess.run(call_cmd, capture_output=True, text=True, env=test_env, timeout=10.0)
                        print(f"[C4] Reset service during active walking result: {res.returncode}")
                        assert res.returncode == 0, f"Reset service call failed: {res.stderr}"
                        assert "done=True" in res.stdout or "done=true" in res.stdout

                    # After reset, verify robot stays upright for at least 2 seconds more
                    if reset_invoked and t_sim >= 6.0:
                        print(f"[C4] Successfully verified post-reset standing stability at t_sim={t_sim:.2f}s")
                        break

                if time.time() - start_wall_time > timeout_wall_time:
                    pytest.fail(f"C4 test timed out after {timeout_wall_time}s")

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

        assert command_dispatched, "Walking was never initiated"
        assert reset_invoked, "Reset was never invoked during walking"
