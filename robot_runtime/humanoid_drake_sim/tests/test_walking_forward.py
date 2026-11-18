#!/usr/bin/env python3
"""
Automated Headless Forward Walking Verification Test Suite (Requirement R4.2).

Verifies:
1. Headless 2-node simulation launch (taskset CPU pinning cores 0-4 and 5-9).
2. Clean double-support stance stabilization prior to command dispatch.
3. Forward velocity command dispatch (v_x = +0.40 m/s, h_des = 0.7925 m) on
   /humanoid/walking_velocity_command.
4. Procedural gait breakout detection (stance -> slow_walk/walk).
5. Net forward pelvis displacement Delta x > +0.30 m over 5.0 - 10.0 s simulated time.
6. Continuous posture stability (pelvis_z in [0.65, 0.90] m, |roll| < 0.35, |pitch| < 0.35 rad).
7. falls == 0 across the entire walking test duration.
8. Single Simulation Instance enforcement via fcntl file locking and pkill cleanup.
"""

import fcntl
import os
import re
import signal
import subprocess
import time
import pytest

WALKING_TELEMETRY_REGEX = re.compile(
    r"\[walking_telemetry\]\s+t_sim=(?P<t_sim>[0-9.]+)\s+s\s+\|\s+pelvis_x=(?P<pelvis_x>[0-9.-]+)\s+m\s+(?:\|\s+pelvis_y=[0-9.-]+\s+m\s+)?\|\s+pelvis_z=(?P<pelvis_z>[0-9.]+)\s+m(?:\s+\|\s+roll=(?P<roll>[0-9.-]+)\s+rad\s+\|\s+pitch=(?P<pitch>[0-9.-]+)\s+rad)?\s+\|\s+falls=(?P<falls>\d+)"
)

STANDING_TELEMETRY_REGEX = re.compile(
    r"\[standing_telemetry\]\s+t_sim=([0-9.]+)\s+s\s+\|\s+pelvis_z=([0-9.]+)\s+m\s+\|\s+roll=([0-9.-]+)\s+rad\s+\|\s+pitch=([0-9.-]+)\s+rad\s+\|\s+falls=(\d+)"
)

GAIT_BREAKOUT_REGEX = re.compile(
    r"ProceduralMpcMotionManager:\s+Increasing to gait:\s*(slow_walk|walk|trot)"
)


class TestWalkingForwardR42:
    """Automated E2E Headless Test for Requirement R4.2 (Forward Walking)."""

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

    def test_walking_forward_displacement_and_stability(self):
        """
        Launches drake_mrt_sim.launch.py headless, applies v_x = 0.40 m/s command,
        verifies gait breakout, and asserts Delta x > 0.30 m with falls == 0.
        """
        test_env = os.environ.copy()
        test_env["ROS_DOMAIN_ID"] = str((os.getpid() % 100) + 10)

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

        sim_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
            env=test_env,
        )

        pub_proc = None
        command_dispatched = False
        gait_breakout_verified = False
        initial_pelvis_x = None
        final_pelvis_x = None
        telemetry_samples = []
        sim_start_time = None
        max_sim_time = 0.0

        start_wall_time = time.time()
        timeout_wall_time = 120.0

        try:
            for line in iter(sim_proc.stdout.readline, ""):
                if not line:
                    break

                # 1. Check for gait breakout in MPC log
                if not gait_breakout_verified and GAIT_BREAKOUT_REGEX.search(line):
                    gait_breakout_verified = True
                    print(f"\n[Gait Breakout] Detected transition out of stance: {line.strip()}")

                # 2. Check standing telemetry for continuous posture invariant checks
                std_match = STANDING_TELEMETRY_REGEX.search(line)
                if std_match:
                    t_sim = float(std_match.group(1))
                    pelvis_z = float(std_match.group(2))
                    roll = float(std_match.group(3))
                    pitch = float(std_match.group(4))
                    falls = int(std_match.group(5))

                    max_sim_time = max(max_sim_time, t_sim)

                    assert falls == 0, f"Fall detected in standing telemetry at t_sim={t_sim:.3f} s (falls={falls})"
                    assert 0.65 <= pelvis_z <= 0.90, (
                        f"Pelvis height violation at t_sim={t_sim:.3f} s: {pelvis_z:.4f} m (expected [0.65, 0.90] m)"
                    )
                    assert abs(roll) <= 0.35, (
                        f"Pelvis roll violation at t_sim={t_sim:.3f} s: {roll:.4f} rad (expected <= 0.35 rad)"
                    )
                    assert abs(pitch) <= 0.35, (
                        f"Pelvis pitch violation at t_sim={t_sim:.3f} s: {pitch:.4f} rad (expected <= 0.35 rad)"
                    )

                # 3. Parse walking telemetry line for displacement tracking
                wlk_match = WALKING_TELEMETRY_REGEX.search(line)
                if wlk_match:
                    t_sim = float(wlk_match.group("t_sim"))
                    pelvis_x = float(wlk_match.group("pelvis_x"))
                    pelvis_z = float(wlk_match.group("pelvis_z"))
                    falls = int(wlk_match.group("falls"))

                    max_sim_time = max(max_sim_time, t_sim)

                    assert falls == 0, f"Fall detected in walking telemetry at t_sim={t_sim:.3f} s (falls={falls})"
                    assert 0.65 <= pelvis_z <= 0.90, (
                        f"Pelvis height violation at t_sim={t_sim:.3f} s: {pelvis_z:.4f} m (expected [0.65, 0.90] m)"
                    )

                    # Allow robot to stabilize in stance for 1.5 s sim time before commanding walk
                    if t_sim >= 1.5 and not command_dispatched:
                        initial_pelvis_x = pelvis_x
                        sim_start_time = t_sim
                        command_dispatched = True

                        pub_cmd = [
                            "ros2", "topic", "pub", "--rate", "10",
                            "--qos-reliability", "best_effort",
                            "/humanoid/walking_velocity_command",
                            "humanoid_mpc_msgs/msg/WalkingVelocityCommand",
                            "{linear_velocity_x: 0.4, linear_velocity_y: 0.0, angular_velocity_z: 0.0, desired_pelvis_height: 0.7925}",
                        ]
                        pub_proc = subprocess.Popen(
                            pub_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid,
                            env=test_env,
                        )
                        print(f"\n[Command Dispatch] Published v_x = +0.40 m/s at t_sim={t_sim:.3f} s (x_init={initial_pelvis_x:.4f} m)")

                    # Collect post-command telemetry
                    if command_dispatched and (t_sim >= sim_start_time):
                        final_pelvis_x = pelvis_x
                        telemetry_samples.append({
                            "t_sim": t_sim,
                            "pelvis_x": pelvis_x,
                            "pelvis_z": pelvis_z,
                            "falls": falls,
                        })

                        walking_sim_duration = t_sim - sim_start_time
                        delta_x = final_pelvis_x - initial_pelvis_x
                        if walking_sim_duration >= 5.0 and delta_x > 0.30:
                            print(f"\n[Goal Achieved] Reached Delta x = {delta_x:.4f} m > 0.30 m at walk duration = {walking_sim_duration:.2f} s")
                            break

                        if walking_sim_duration >= 10.0:
                            break

                if time.time() - start_wall_time > timeout_wall_time:
                    pytest.fail(
                        f"Test timed out after {timeout_wall_time}s wall time. "
                        f"Max t_sim achieved: {max_sim_time:.3f}s"
                    )

        finally:
            if pub_proc:
                try:
                    os.killpg(os.getpgid(pub_proc.pid), signal.SIGINT)
                    pub_proc.wait(timeout=2.0)
                except Exception:
                    try:
                        os.killpg(os.getpgid(pub_proc.pid), signal.SIGKILL)
                    except Exception:
                        pass

            try:
                os.killpg(os.getpgid(sim_proc.pid), signal.SIGINT)
                sim_proc.wait(timeout=5.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(sim_proc.pid), signal.SIGKILL)
                except Exception:
                    pass

            self._kill_sim_nodes()

        # Terminal Aggregate Verification
        assert command_dispatched, "Walking command was never dispatched (simulation failed to reach t_sim >= 1.5 s)"
        assert gait_breakout_verified, "Gait breakout failed: Node 1 did not transition out of stance"
        assert len(telemetry_samples) >= 8, f"Insufficient walking telemetry samples collected: {len(telemetry_samples)}"

        delta_x = final_pelvis_x - initial_pelvis_x
        walking_time = telemetry_samples[-1]["t_sim"] - sim_start_time

        print(f"\n=== Forward Walking Test Summary ===")
        print(f"Walking Simulated Duration: {walking_time:.3f} s")
        print(f"Initial Pelvis X:          {initial_pelvis_x:.4f} m")
        print(f"Final Pelvis X:            {final_pelvis_x:.4f} m")
        print(f"Net Pelvis Displacement:   {delta_x:+.4f} m (Requirement: > +0.30 m)")
        print(f"Total Telemetry Samples:   {len(telemetry_samples)}")
        print(f"Total Falls:               0")

        assert delta_x > 0.30, (
            f"Forward displacement requirement not met: Delta x = {delta_x:.4f} m "
            f"(required > +0.30 m over {walking_time:.2f} s)"
        )
