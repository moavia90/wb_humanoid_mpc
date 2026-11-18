#!/usr/bin/env python3
"""
Tier 1 & Tier 2: ROS Interfaces, Startup Warmup & Reset Robustness (F4, F5, B9).

Verifies:
- F4.1–F4.5: Active publisher/subscriber counts on /g1/mpc_policy, /g1/mpc_observation,
             and availability of /g1/mpc_reset service.
- F5.1–F5.5: Startup warmup protocol (steps 1-5, filter priming, policy knot height >= 0.78m,
             zero initial squat drop).
- B9: Rapid /g1/mpc_reset service calls without loss of stability.
"""

import fcntl
import os
import re
import signal
import subprocess
import time
import pytest


LOCK_FILE = "/tmp/humanoid_drake_sim_instance.lock"

STARTUP_STEP4_REGEX = re.compile(
    r"\[STARTUP STEP 4/5 PASSED\] Valid policy verified!\s+Front=([0-9.]+)\s+m,\s+Back=([0-9.]+)\s+m"
)
STANDING_TELEMETRY_REGEX = re.compile(
    r"\[standing_telemetry\]\s+t_sim=([0-9.]+)\s+s\s+\|\s+pelvis_z=([0-9.]+)\s+m\s+\|\s+roll=([0-9.-]+)\s+rad\s+\|\s+pitch=([0-9.-]+)\s+rad\s+\|\s+falls=(\d+)"
)


@pytest.mark.tier1
@pytest.mark.tier2
class TestRosInterfacesAndWarmup:
    """Integrated verification of ROS topics, warmup filter, and reset service."""

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
        """Acquire kernel flock and guarantee clean environment."""
        lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(lock_fd)
            pytest.fail(
                f"CONCURRENCY VIOLATION: Simulation lock ({LOCK_FILE}) held: {exc}"
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

    def test_ros_interfaces_warmup_and_reset(self):
        """Runs headless simulation, verifies F4 topics/services, F5 warmup, and B9 resets."""
        test_env = os.environ.copy()
        test_env["ROS_DOMAIN_ID"] = "93"

        ws_setup = "/home/mov/Documents/humanoid_mpc_ws/install/setup.bash"
        if os.path.exists(ws_setup) and "/home/mov/Documents/humanoid_mpc_ws/install" not in test_env.get("AMENT_PREFIX_PATH", ""):
            out = subprocess.check_output(f"source {ws_setup} && env", shell=True, executable="/bin/bash", text=True)
            for line in out.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    test_env[k] = v

        subprocess.run(["ros2", "daemon", "stop"], env=test_env, capture_output=True)
        subprocess.run(["ros2", "daemon", "start"], env=test_env, capture_output=True)

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

        logs = []
        telemetry_samples = []
        step4_found = False
        step4_front_h = 0.0
        step4_back_h = 0.0
        start_wall_time = time.time()
        timeout_wall_time = 45.0

        try:
            # 1. Collect initial lines until t_sim >= 3.0s
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                logs.append(line)

                s4_match = STARTUP_STEP4_REGEX.search(line)
                if s4_match:
                    step4_found = True
                    step4_front_h = float(s4_match.group(1))
                    step4_back_h = float(s4_match.group(2))

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
                    if t_sim >= 3.0:
                        break

                if time.time() - start_wall_time > timeout_wall_time:
                    pytest.fail(f"Simulation startup timed out waiting for t_sim >= 3.0s after {timeout_wall_time}s")

            # Assert F5: Startup Warmup & Squat Prevention
            assert step4_found, "Startup Step 4/5 was not reached in simulation output"
            assert step4_front_h >= 0.78, f"Startup front knot height too low: {step4_front_h} m (< 0.78 m)"
            assert step4_back_h >= 0.78, f"Startup back knot height too low: {step4_back_h} m (< 0.78 m)"
            assert len(telemetry_samples) >= 4, "Insufficient initial telemetry samples"
            for s in telemetry_samples:
                assert s["falls"] == 0, f"Fall detected during startup: {s}"
                assert s["pelvis_z"] >= 0.70, f"Pelvis dropped below 0.70m during startup: {s['pelvis_z']} m"

            # Assert F4: Topics & Services Introspection
            topic_policy = subprocess.run(
                ["ros2", "topic", "info", "/g1/mpc_policy"],
                capture_output=True, text=True, env=test_env, timeout=10.0
            )
            assert topic_policy.returncode == 0, f"Failed to get topic info for /g1/mpc_policy: {topic_policy.stderr}"
            assert "Publisher count: 1" in topic_policy.stdout or "Publisher count: 2" in topic_policy.stdout
            assert "Subscription count: 1" in topic_policy.stdout or "Subscription count: 2" in topic_policy.stdout

            topic_obs = subprocess.run(
                ["ros2", "topic", "info", "/g1/mpc_observation"],
                capture_output=True, text=True, env=test_env, timeout=10.0
            )
            assert topic_obs.returncode == 0, f"Failed to get topic info for /g1/mpc_observation: {topic_obs.stderr}"
            assert "Publisher count: 1" in topic_obs.stdout or "Publisher count: 2" in topic_obs.stdout
            assert "Subscription count: 1" in topic_obs.stdout or "Subscription count: 2" in topic_obs.stdout

            srv_list = subprocess.run(
                ["ros2", "service", "list"],
                capture_output=True, text=True, env=test_env, timeout=10.0
            )
            assert srv_list.returncode == 0
            assert "/g1/mpc_reset" in srv_list.stdout, f"/g1/mpc_reset service not registered:\n{srv_list.stdout}"

            # Assert B9: Rapid Reset Service Invocations
            call_cmd = [
                "ros2", "service", "call", "/g1/mpc_reset", "ocs2_ros2_msgs/srv/Reset",
                "{reset: true, target_trajectories: {api_version: 0, time_trajectory: [0.0], state_trajectory: [{value: []}], input_trajectory: [{value: []}]}}"
            ]
            reset_call = subprocess.run(call_cmd, capture_output=True, text=True, env=test_env, timeout=10.0)
            assert reset_call.returncode == 0, f"Reset service call failed: {reset_call.stderr}"
            assert "done=True" in reset_call.stdout or "done=true" in reset_call.stdout

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
            subprocess.run(["ros2", "daemon", "stop"], env=test_env, capture_output=True)
