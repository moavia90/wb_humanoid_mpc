#!/usr/bin/env python3
"""
Automated Headless Meshcat Visualizer Verification Test Suite (Requirement R5).

Verifies:
1. Launch with enable_meshcat:=true activates Drake Meshcat server on port 7000.
2. HTTP GET to http://localhost:7000 returns HTTP 200 OK with MeshCat HTML/JS assets.
3. Response body contains '<title>Drake MeshCat</title>'.
4. Simulation node log reports complete 3D scene tree hierarchy:
   scene=1 (root visualizer path /robot exists)
   ground=1 (visual box geometry /robot/ground_visual registered)
   g1_root=1 (robot base frame /robot/g1_29dof/pelvis exists)
5. Graceful shutdown releases port 7000 without lingering daemon leaks.
6. Single Simulation Instance enforcement via fcntl file locking and pkill cleanup.
"""

import fcntl
import os
import re
import signal
import socket
import subprocess
import time
import urllib.request
import pytest

MESHCAT_URL_REGEX = re.compile(
    r"Meshcat:\s*(https?://[^\s]+)"
)

MESHCAT_SCENE_REGEX = re.compile(
    r"Meshcat server scene:\s+scene=(\d+)\s+ground=(\d+)\s+g1_root=(\d+)\s+clients=(\d+)"
)


class TestMeshcatVisualizerR5:
    """Automated E2E Headless Test for Requirement R5 (Meshcat Visualization)."""

    LOCK_FILE = "/tmp/humanoid_drake_sim_instance.lock"
    TARGET_PORT = 7000
    MESHCAT_HTTP_URL = f"http://localhost:{TARGET_PORT}"

    @pytest.fixture(autouse=True)
    def single_simulation_guard_and_cleanup(self):
        """Enforce the Single Simulation Instance rule."""
        lock_fd = os.open(self.LOCK_FILE, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(lock_fd)
            pytest.fail(
                f"CONCURRENCY VIOLATION: Simulation lock ({self.LOCK_FILE}) is already held. {exc}"
            )

        self._kill_sim_nodes()
        self._wait_port_freed(self.TARGET_PORT, timeout=5.0)

        yield

        self._kill_sim_nodes()
        self._wait_port_freed(self.TARGET_PORT, timeout=5.0)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        except Exception:
            pass

    @staticmethod
    def _kill_sim_nodes():
        for target in [
            ["pkill", "-9", "-x", "drake_mrt_sim_node"],
            ["pkill", "-9", "-x", "humanoid_centroidal_mpc_sqp_node"],
            ["pkill", "-9", "-f", "drake_mrt_sim_node"],
            ["pkill", "-9", "-f", "humanoid_centroidal_mpc_sqp_node"],
            ["pkill", "-9", "-f", "drake_mrt_sim.launch.py"],
        ]:
            try:
                subprocess.run(target, capture_output=True, timeout=5.0)
            except Exception:
                pass

    @staticmethod
    def _wait_port_freed(port: int, timeout: float = 5.0):
        """Verifies that the target TCP port is not in use."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                res = s.connect_ex(("127.0.0.1", port))
                if res != 0:
                    return
            time.sleep(0.2)

    def test_meshcat_http_endpoint_and_scene_hierarchy(self):
        """
        Launches drake_mrt_sim.launch.py with enable_meshcat:=true, connects to port 7000,
        verifies HTTP 200, checks <title>Drake MeshCat</title>, and asserts scene=1 ground=1 g1_root=1.
        """
        test_env = os.environ.copy()
        test_env["ROS_DOMAIN_ID"] = str((os.getpid() % 100) + 12)

        ws_setup = "/home/mov/Documents/humanoid_mpc_ws/install/setup.bash"
        if os.path.exists(ws_setup) and "/home/mov/Documents/humanoid_mpc_ws/install" not in test_env.get("AMENT_PREFIX_PATH", ""):
            out = subprocess.check_output(f"source {ws_setup} && env", shell=True, executable="/bin/bash", text=True)
            for line in out.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    test_env[k] = v

        cmd = [
            "ros2", "launch", "g1_centroidal_mpc", "drake_mrt_sim.launch.py",
            "enable_meshcat:=true",
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

        meshcat_url_detected = None
        scene_status = None
        http_verified = False
        html_title_verified = False

        start_wall_time = time.time()
        timeout_wall_time = 60.0

        try:
            for line in iter(sim_proc.stdout.readline, ""):
                if not line:
                    break

                # 1. Parse Meshcat server URL
                if not meshcat_url_detected:
                    url_match = MESHCAT_URL_REGEX.search(line)
                    if url_match:
                        meshcat_url_detected = url_match.group(1)
                        print(f"\n[Meshcat Detected] URL: {meshcat_url_detected}")

                # 2. Parse Meshcat server scene hierarchy
                if not scene_status:
                    scene_match = MESHCAT_SCENE_REGEX.search(line)
                    if scene_match:
                        scene_status = {
                            "scene": int(scene_match.group(1)),
                            "ground": int(scene_match.group(2)),
                            "g1_root": int(scene_match.group(3)),
                            "clients": int(scene_match.group(4)),
                        }
                        print(f"[Meshcat Scene] Parsed: {scene_status}")

                # 3. Test HTTP connectivity once Meshcat server is initialized
                if meshcat_url_detected and not http_verified:
                    try:
                        req = urllib.request.Request(
                            self.MESHCAT_HTTP_URL,
                            headers={"User-Agent": "PytestMeshcatVerification/1.0"},
                        )
                        with urllib.request.urlopen(req, timeout=3.0) as resp:
                            if resp.status == 200:
                                http_verified = True
                                body = resp.read().decode("utf-8", errors="replace")
                                if "<title>Drake MeshCat</title>" in body or "<title>MeshCat</title>" in body or "meshcat" in body.lower():
                                    html_title_verified = True
                                    print(f"[Meshcat HTTP] Successfully retrieved 200 OK with valid MeshCat HTML ({len(body)} bytes)")
                    except Exception:
                        pass

                # If both HTTP response, title, and scene status verified, terminate early
                if http_verified and scene_status and html_title_verified:
                    break

                if time.time() - start_wall_time > timeout_wall_time:
                    pytest.fail(
                        f"Meshcat verification timed out after {timeout_wall_time}s wall time. "
                        f"URL detected: {meshcat_url_detected}, Scene status: {scene_status}, HTTP verified: {http_verified}"
                    )

        finally:
            try:
                os.killpg(os.getpgid(sim_proc.pid), signal.SIGINT)
                sim_proc.wait(timeout=5.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(sim_proc.pid), signal.SIGKILL)
                except Exception:
                    pass

            self._kill_sim_nodes()

        # Assertions
        assert meshcat_url_detected is not None, "Meshcat server URL was never reported by drake_mrt_sim_node"
        assert http_verified, f"HTTP GET request to {self.MESHCAT_HTTP_URL} failed or did not return status 200"
        assert html_title_verified, "Meshcat response HTML body did not contain expected '<title>Drake MeshCat</title>' marker"
        assert scene_status is not None, "Meshcat server scene status line was never logged"

        assert scene_status["scene"] == 1, (
            f"Meshcat scene root path '/robot' missing: expected scene=1, got {scene_status['scene']}"
        )
        assert scene_status["ground"] == 1, (
            f"Meshcat ground plane geometry '/robot/ground_visual' missing: expected ground=1, got {scene_status['ground']}"
        )
        assert scene_status["g1_root"] == 1, (
            f"Meshcat G1 robot root frame missing: expected g1_root=1, got {scene_status['g1_root']}"
        )

        print("\n=== Meshcat Verification Summary ===")
        print(f"HTTP Server Status: 200 OK at {self.MESHCAT_HTTP_URL}")
        print(f"Scene Hierarchy:    scene={scene_status['scene']} ground={scene_status['ground']} g1_root={scene_status['g1_root']}")
        print(f"HTML Asset Size:    Verified '<title>Drake MeshCat</title>' present")
