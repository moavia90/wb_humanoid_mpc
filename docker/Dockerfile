ARG ROS_DISTRO=jazzy
ARG BASE_IMAGE=ros:${ROS_DISTRO}-ros-base
FROM ${BASE_IMAGE} AS base

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG WB_HUMANOID_MPC_DIR="/wb_humanoid_mpc_ws"
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=${ROS_DISTRO}

# Install system dependencies, ROS, Python, and development tools in a single step
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl coreutils gnupg2 lsb-release software-properties-common \
    locales \
    git \
    x11-apps \
    mesa-utils \
    gettext-base \
    && rm -rf /var/lib/apt/lists/*

COPY dependencies.txt /tmp/dependencies.txt

RUN apt-get update && \
    envsubst < /tmp/dependencies.txt | xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gnupg wget terminator \
    && rm -rf /var/lib/apt/lists/*
RUN wget -qO- https://drake-apt.csail.mit.edu/drake.asc | gpg --dearmor - \
    | tee /etc/apt/trusted.gpg.d/drake.gpg >/dev/null

RUN echo "deb [arch=amd64] https://drake-apt.csail.mit.edu/$(lsb_release -cs) $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/drake.list >/dev/null

RUN apt-get update && apt-get install -y --no-install-recommends drake-dev && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/drake/bin${PATH:+:${PATH}}"
ENV PYTHONPATH="/opt/drake/lib/python$(python3 -c 'import sys; print("{0}.{1}".format(*sys.version_info))')/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

# Clone drake-ros and install prerequisites using their script
WORKDIR /tmp
RUN git clone https://github.com/RobotLocomotion/drake-ros.git /tmp/drake-ros

# Use drake-ros install script to install dependencies (following the https://github.com/RobotLocomotion/drake-ros/blob/main/.devcontainer/Dockerfile)
RUN yes | /tmp/drake-ros/setup/install_prereqs \
    -C /tmp/throwaway --with-colcon && \
    rm -rf /tmp/throwaway /var/lib/apt/lists/*

# Set up drake-ros workspace
WORKDIR /drake_ros_ws
RUN mkdir -p src
RUN mv /tmp/drake-ros src/drake-ros

# Build drake_ros packages
RUN /bin/bash -c "\
  source /opt/ros/${ROS_DISTRO}/setup.bash && \
  rosdep update && \
  rosdep install --from-paths src --ignore-src -r -y && \
  colcon build --packages-select drake_ros drake_ros_examples --symlink-install \
    --cmake-args \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS='-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized' && \
       source install/setup.bash \
"
# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Set up environment variables
ENV CMAKE_PREFIX_PATH="/opt/ros/${ROS_DISTRO}:/drake_ros_ws/install:${CMAKE_PREFIX_PATH}"
ENV DISPLAY=${DISPLAY} QT_X11_NO_MITSHM=1

# Set up user and workspace
WORKDIR ${WB_HUMANOID_MPC_DIR}
RUN mkdir -p ${WB_HUMANOID_MPC_DIR}/src/wb_humanoid_mpc

# Ensure correct UID/GID
RUN if ! getent group $GROUP_ID > /dev/null; then groupadd --gid $GROUP_ID devgroup; fi && \
    if ! getent passwd $USER_ID > /dev/null; then useradd --uid $USER_ID --gid $GROUP_ID --create-home devuser; fi && \
    chown -R $USER_ID:$GROUP_ID .

# Copy the Makefile and other necessary files
COPY . ${WB_HUMANOID_MPC_DIR}/src/wb_humanoid_mpc
