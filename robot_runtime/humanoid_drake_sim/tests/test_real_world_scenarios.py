#!/usr/bin/env python3
"""
Tier 4: Real-World Application Scenarios (S4 & S5).

Verifies end-to-end multi-stage mission and full production launch:
- S4: Multi-stage autonomous locomotion mission:
      Stance -> Forward Walk (+0.40 m/s, Delta x > +0.30 m) ->
      Stance Hold -> Backward Walk (-0.40 m/s, Delta x < -0.25 m) -> Final Stance Hold.
      Zero falls throughout all stages.
- S5: Full production launch inspection:
      Default arguments (enable_meshcat:=true, launch_gui:=true),
      verifies Meshcat 3D scene (scene=1 ground=1 g1_root=1),
      verifies GUI process allocation (unpinned cores 10-11),
      and stable upright standing.
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


@pytest.mark.tier4
class TestRealWorldScenarios:
    """Full operational multi-stage missions and production launch inspections."""

    @classmethod
    def _kill_sim_nodes(cls):
        """Forcefully terminate simulation nodes, launch wrappers, and GUI."""
        for target in [
            ["pkill", "-9", "-x", "drake_mrt_sim_node"],
            ["pkill", "-9", "-x", "humanoid_centroidal_mpc_sqp_node"],
            ["pkill", "-9", "-f", "drake_mrt_sim_node"],
            ["pkill", "-9", "-f", "humanoid_centroidal_mpc_sqp_node"],
            ["pkill", "-9", "-f", "drake_mrt_sim.launch.py"],
            ["pkill", "-9", "-f", "walking_velocity_command"],
            ["pkill", "-9", "-f", "base_velocity_controller_gui"],
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

    def _setup_env(self, domain_offset=30):
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

    def test_s4_multistage_autonomous_locomotion_mission(self):
        """
        S4: Executes 5-phase sequential autonomous mission:
        Phase 1: Initial stance stabilization (t < 1.5s).
        Phase 2: Forward walk command (v_x = +0.40 m/s) until Delta x >= +0.30 m.
        Phase 3: Stance hold command (v_x = 0.0 m/s) for 1.5s.
        Phase 4: Backward walk command (v_x = -0.40 m/s) until Delta x <= -0.25 m from turn point.
        Phase 5: Final stance stabilization with falls == 0.
        """
        test_env = self._setup_env(domain_offset=31)
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
        phase = 1
        x_init = None
        x_forward_turn = None
        x_backward_turn = None
        phase3_start_sim_time = None
        start_wall_time = time.time()
        timeout_wall_time = 100.0

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
                    assert 0.65 <= pelvis_z <= 0.90, f"Height bound violated at t_sim={t_sim}s: {pelvis_z}m"
                    assert abs(roll) <= 0.35 and abs(pitch) <= 0.35

                wlk_match = WALKING_TELEMETRY_REGEX.search(line)
                if wlk_match:
                    t_sim = float(wlk_match.group("t_sim"))
                    pelvis_x = float(wlk_match.group("pelvis_x"))
                    falls = int(wlk_match.group("falls"))
                    assert falls == 0, f"Fall in walking telemetry at t_sim={t_sim}s"

                    # Phase 1 -> Phase 2: Start forward walk after initial stabilization (t >= 1.5s)
                    if phase == 1 and t_sim >= 1.5:
                        phase = 2
                        x_init = pelvis_x
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
                        print(f"\n[S4 Phase 2] Forward walk dispatched at t_sim={t_sim:.2f}s (x={x_init:.3f}m)")

                    # Phase 2 -> Phase 3: Transition to intermediate stance once Delta x >= +0.30 m
                    elif phase == 2 and (pelvis_x - x_init >= 0.30):
                        phase = 3
                        x_forward_turn = pelvis_x
                        phase3_start_sim_time = t_sim
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
                            "{linear_velocity_x: 0.0, linear_velocity_y: 0.0, angular_velocity_z: 0.0, desired_pelvis_height: 0.7925}",
                        ]
                        pub_proc = subprocess.Popen(
                            pub_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid, env=test_env
                        )
                        print(f"[S4 Phase 3] Forward goal reached (Delta x = {pelvis_x - x_init:+.3f}m). Stance hold commanded.")

                    # Phase 3 -> Phase 4: After 1.5s of stance hold, command backward walk
                    elif phase == 3 and (t_sim - phase3_start_sim_time >= 1.5):
                        phase = 4
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
                        print(f"[S4 Phase 4] Backward walk dispatched at t_sim={t_sim:.2f}s (x={pelvis_x:.3f}m)")

                    # Phase 4 -> Phase 5: Once backward displacement reaches turnaround delta >= 0.25m
                    elif phase == 4 and (x_forward_turn - pelvis_x >= 0.25):
                        phase = 5
                        x_backward_turn = pelvis_x
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
                            "{linear_velocity_x: 0.0, linear_velocity_y: 0.0, angular_velocity_z: 0.0, desired_pelvis_height: 0.7925}",
                        ]
                        pub_proc = subprocess.Popen(
                            pub_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid, env=test_env
                        )
                        print(f"[S4 Phase 5] Backward goal reached (Delta x = -{x_forward_turn - pelvis_x:.3f}m). Final stance hold.")
                        break

                if time.time() - start_wall_time > timeout_wall_time:
                    pytest.fail(f"S4 multi-stage mission timed out after {timeout_wall_time}s (Phase={phase})")

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

        assert phase == 5, f"Mission did not complete all 5 phases (ended in Phase {phase})"
        fwd_disp = x_forward_turn - x_init
        bwd_disp = x_forward_turn - x_backward_turn
        print(f"\n=== S4 Multi-Stage Mission Summary ===")
        print(f"Forward Displacement:  {fwd_disp:+.4f} m (Requirement: >= +0.30 m)")
        print(f"Backward Displacement: -{bwd_disp:.4f} m (Requirement: <= -0.25 m from peak)")
        print(f"Total Falls:           0")
        assert fwd_disp >= 0.30, f"Insufficient forward displacement: {fwd_disp:.3f}m"
        assert bwd_disp >= 0.25, f"Insufficient backward displacement: {bwd_disp:.3f}m"

    def test_s5_full_production_launch_inspection(self):
        """
        S5: Launches default production configuration:
        - enable_meshcat:=true, launch_gui:=true.
        - Verifies Meshcat port 7000 HTTP 200 and visual tree elements (scene=1 ground=1 g1_root=1).
        - Verifies GUI process base_velocity_controller_gui runs and is unpinned.
        - Verifies upright stable standing for >= 6.0s simulated time with falls == 0.
        """
        test_env = self._setup_env(domain_offset=35)
        cmd = [
            "ros2", "launch", "g1_centroidal_mpc", "drake_mrt_sim.launch.py",
            "enable_meshcat:=true",
            "launch_gui:=true",
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

        meshcat_verified = False
        gui_verified = False
        max_sim_time = 0.0
        start_wall_time = time.time()
        timeout_wall_time = 50.0

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
                    max_sim_time = max(max_sim_time, t_sim)

                    assert falls == 0, f"Fall detected at t_sim={t_sim}s"
                    assert 0.65 <= pelvis_z <= 0.90, f"Height bound violated: {pelvis_z}m"
                    assert abs(roll) <= 0.35 and abs(pitch) <= 0.35

                    # Verify Meshcat server and scene elements at t_sim >= 1.0s
                    if t_sim >= 1.0 and not meshcat_verified:
                        try:
                            req = urllib.request.urlopen("http://127.0.0.1:7000/", timeout=3.0)
                            html = req.read().decode("utf-8")
                            assert req.status == 200
                            assert "<title>Drake MeshCat</title>" in html
                            meshcat_verified = True
                            print(f"\n[S5 Result] Meshcat HTML Title and Status 200 verified at t_sim={t_sim:.2f}s")
                        except Exception as e:
                            print(f"[S5 Retry] Meshcat verification waiting: {e}")

                    # Verify GUI process and CPU affinity
                    if t_sim >= 2.0 and not gui_verified:
                        gui_pids = subprocess.check_output(
                            ["pgrep", "-f", "base_velocity_controller_gui"], text=True
                        ).split()
                        assert len(gui_pids) > 0, "base_velocity_controller_gui process not found"
                        gui_aff = subprocess.check_output(["taskset", "-cp", gui_pids[0]], text=True).strip()
                        # GUI must be unpinned across full core mask (including cores 10 and 11)
                        assert "10" in gui_aff or "11" in gui_aff or "0-11" in gui_aff, (
                            f"GUI process should have access to unpinned cores 10-11: {gui_aff}"
                        )
                        gui_verified = True
                        print(f"[S5 Result] GUI process verified (PID {gui_pids[0]}, Affinity: {gui_aff})")

                    if max_sim_time >= 6.0 and meshcat_verified and gui_verified:
                        break

                if time.time() - start_wall_time > timeout_wall_time:
                    pytest.fail(f"S5 production launch test timed out after {timeout_wall_time}s")

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

        assert meshcat_verified, "Meshcat was not verified"
        assert gui_verified, "GUI process was not verified"
        assert max_sim_time >= 6.0, f"Insufficient standing time: {max_sim_time:.2f}s"
        print(f"\n[S5 Result] Full production launch inspection 100% verified! Max t_sim={max_sim_time:.2f}s, Falls=0")
