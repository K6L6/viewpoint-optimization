FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
MAINTAINER Kelvin

ENV DEBIAN_FRONTEND noninteractive

# Packages for VNC support. referred:https://github.com/ct2034/docker-ubuntu-vnc-desktop/blob/master/Dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common curl
RUN add-apt-repository ppa:fcwu-tw/ppa
RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
	supervisor \
	openssh-server pwgen sudo vim-tiny \
	net-tools \
	lxde x11vnc xvfb \
	gtk2-engines-murrine ttf-ubuntu-font-family \
	firefox \
	nginx \
	python3 python3-pip python3-dev build-essential \
	mesa-utils libgl1-mesa-dri \
	gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine pinta \
	dbus-x11 x11-utils \
	terminator \
	gedit \
	okular \
	vim \
	wget \
	git-core \
	apt-utils \
	python3-setuptools \
	lsb-release \
	&& apt-get autoclean \
	&& apt-get autoremove \
	&& rm -rf /var/lib/apt/lists/*

# install ros (source: https://github.com/osrf/docker_images/blob/5399f380af0a7735405a4b6a07c6c40b867563bd/ros/kinetic/ubuntu/xenial/ros-core/Dockerfile)
# install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*
	


#BUilding ROS Kinetic with python3 support
RUN pip3 install --upgrade pip && pip3 install rosdep \
				rospkg \
				rosinstall_generator \
				rosinstall \
				wstool \
				vcstools \
				catkin_tools \
				catkin_pkg

# Dependencies referred: http://sdk.rethinkrobotics.com/wiki/Simulator_Installation
RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 
RUN apt-get update && apt-get install -y ros-kinetic-desktop-full

# from https://pillow.readthedocs.io/en/stable/installation.html
RUN apt-get install -y libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    libharfbuzz-dev libfribidi-dev
#     && sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list' \
#     && wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add - 
#     && apt-get update \
#     && apt-get install -y python-rosinstall \
# 			# ros-kinetic-ros-base \ # use when gazebo version over 7 needed.
# 			ros-kinetic-desktop-full \
# 			# git-core \
# 			# python-argparse \
# 			# python-wstool \
# 			# python-vcstools \
# 			# python-rosdep \
# 			# python-catkin-tools \
# 			# ros-kinetic-control-msgs \
# 			ros-kinetic-joystick-drivers \
# 			&& rm -rf /var/lib/apt/lists/*
# gazebo7 libgazebo7-dev \

# setup environment
# ENV LANG C.UTF-8
# ENV LC_ALL C.UTF-8

RUN rosdep init && rosdep update
# RUN apt-get update && apt-get install -y python3-catkin-pkg-modules \
# 										python3-rospkg-modules \
# 										python-wxtools
# RUN apt-get update && apt-get install -y ros-kinetic-qt-build \
# 			ros-kinetic-gazebo-ros-control \
# 			# ros-kinetic-gazebo-ros-pkgs \
# 			# ros-kinetic-gazebo9* \
# 			ros-kinetic-ros-control \
# 			# ros-kinetic-control-toolbox \
# 			# ros-kinetic-robot-state-publisher \
# 			# ros-kinetic-realtime-tools \
# 			ros-kinetic-ros-controllers \
# 			# ros-kinetic-xacro \
# 			# ros-kinetic-tf-conversions \
# 			# ros-kinetic-kdl-parser \
# 			gazebo7 \
# 			libgazebo7-dev \
# 			&& rm -rf /var/lib/apt/lists/*

# ROS with python3 setup
ENV CATKIN_WS=/home/baxter_ws
RUN mkdir -p $CATKIN_WS/src
WORKDIR $CATKIN_WS
RUN catkin config --init -DCMAKE_BUILD_TYPE=Release --blacklist rqt_rviz rviz_plugin_tutorials librviz_tutorial --install

RUN rosinstall_generator desktop_full --rosdistro kinetic --deps --tar > kinetic-desktop-full.rosinstall
RUN wstool init -j8 src kinetic-desktop-full.rosinstall
RUN wstool update -j4 -t src

RUN export ROS_PYTHON_VERSION=3 \
	&& pip3 install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04 wxPython
COPY install_skip ./
RUN chmod +x ./install_skip 
RUN ./install_skip `rosdep check --from-paths src --ignore-src | grep python | sed -e "s/^apt\t//g" | sed -z "s/\n/ /g" | sed -e "s/python-/python3-/g"` 
RUN rosdep install --from-paths src --ignore-src -y --skip-keys="`rosdep check --from-paths src --ignore-src | grep python | sed -e "s/^apt\t//g" | sed -z "s/\n/ /g"`" 
RUN find . -type f -exec sed -i 's/\/usr\/bin\/env[ ]*python/\/usr\/bin\/env python3/g' {} +

RUN catkin build \
	&& export PYTHONPATH=/usr/lib/python3/dist-packages \
	&& source $CATKIN_WS/install/setup.bash

# # # # # # # # # # 
# # Baxter Setup #
# # # # # # # # # 

# # workspace & hardware workstation setup
# WORKDIR $CATKIN_WS/src
# RUN /bin/bash -c 'source /opt/ros/kinetic/setup.bash; cd ..; catkin_make; catkin_make install'

# WORKDIR $CATKIN_WS/src
# # Download source code
# RUN wstool init . && \
#     wstool merge https://raw.githubusercontent.com/RethinkRobotics/baxter/master/baxter_sdk.rosinstall && \
#     wstool update
# RUN /bin/bash -c 'source /opt/ros/kinetic/setup.bash; cd ..; catkin_make; catkin_make install'

# WORKDIR $CATKIN_WS

# RUN wget https://github.com/RethinkRobotics/baxter/raw/master/baxter.sh
# RUN chmod u+x baxter.sh

# #simulator setup
# WORKDIR $CATKIN_WS/src
# RUN wstool merge https://raw.githubusercontent.com/RethinkRobotics/baxter_simulator/kinetic-devel/baxter_simulator.rosinstall -y && \
#     wstool update
# RUN /bin/bash -c 'source /opt/ros/kinetic/setup.bash; cd ..; catkin_make'

# # kinetic moveit setup
# WORKDIR $CATKIN_WS/src
# RUN git clone https://github.com/ros-planning/moveit_robots.git \
# 	&& apt-get update \
# 	&& apt-get install ros-kinetic-moveit -y
# # cd ~/baxter_ws/ | ./baxter.sh | catkin_make

# # # # # # # # # # # # # # #
# # tini subreap for VNC? #
# # # # # # # # # # # # #

# ENV TINI_VERSION v0.9.0
# ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
# RUN chmod +x /bin/tini

# ADD image /
# RUN pip install setuptools wheel && pip install -r /usr/lib/web/requirements.txt

# RUN cp /usr/share/applications/terminator.desktop /root/Desktop
# RUN echo "source /opt/ros/kinetic/setup.bash" >> /root/.bashrc

# EXPOSE 80
# WORKDIR /root
# ENV HOME=/home/ubuntu \
#     SHELL=/bin/bash
# ENTRYPOINT ["/startup.sh"]