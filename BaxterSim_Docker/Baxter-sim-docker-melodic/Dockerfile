FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
MAINTAINER Kelvin

ENV DEBIAN_FRONTEND noninteractive

# Packages for VNC support. referred:https://github.com/ct2034/docker-ubuntu-vnc-desktop/blob/master/Dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common curl
#RUN add-apt-repository ppa:fcwu-tw/ppa
RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
	supervisor \
	openssh-server pwgen sudo vim-tiny \
	net-tools \
	lxde x11vnc xvfb \
	gtk2-engines-murrine ttf-ubuntu-font-family \
	firefox \
	nginx \
	python-pip python-dev build-essential \
	mesa-utils libgl1-mesa-dri \
	gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine pinta \
	dbus-x11 x11-utils \
	terminator \
	gedit \
	okular \
	vim \
	wget \
	lsb-release \
	&& apt-get autoclean \
	&& apt-get autoremove \
	&& rm -rf /var/lib/apt/lists/*

# install ros (source: https://github.com/osrf/docker_images/blob/5399f380af0a7735405a4b6a07c6c40b867563bd/ros/melodic/ubuntu/xenial/ros-core/Dockerfile)
# install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies referred: http://sdk.rethinkrobotics.com/wiki/Simulator_Installation
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
    && sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list' \
    && wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add - \
    && apt-get update \
    && apt-get install -y python-rosinstall \
			# ros-melodic-ros-base \ # use when gazebo version over 7 needed.
			ros-melodic-desktop-full \
			# git-core \
			# python-argparse \
			# python-wstool \
			# python-vcstools \
			# python-rosdep \
			# python-catkin-tools \
			# ros-melodic-control-msgs \
			ros-melodic-joystick-drivers \
			&& rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN rosdep init && rosdep update
RUN apt-get update && apt-get install -y ros-melodic-gazebo-ros-control \
			#ros-melodic-qt-build \
			# ros-melodic-gazebo-ros-pkgs \
			# ros-melodic-gazebo9* \
			ros-melodic-ros-control \
			# ros-melodic-control-toolbox \
			# ros-melodic-robot-state-publisher \
			# ros-melodic-realtime-tools \
			ros-melodic-ros-controllers \
			# ros-melodic-xacro \
			# ros-melodic-tf-conversions \
			# ros-melodic-kdl-parser \
			gazebo9 \
			libgazebo9-dev \
			&& rm -rf /var/lib/apt/lists/*


# # # # # # # # # 
# Baxter Setup #
# # # # # # # # 

# workspace & hardware workstation setup
ENV CATKIN_WS=/home/baxter_ws
RUN mkdir -p $CATKIN_WS/src
WORKDIR $CATKIN_WS/src

RUN /bin/bash -c 'source /opt/ros/melodic/setup.bash; cd ..; catkin_make; catkin_make install'

WORKDIR $CATKIN_WS/src
# Download source code
RUN wstool init . && \
    wstool merge https://raw.githubusercontent.com/RethinkRobotics/baxter/master/baxter_sdk.rosinstall && \
    wstool update
RUN /bin/bash -c 'source /opt/ros/melodic/setup.bash; cd ..; catkin_make; catkin_make install'

WORKDIR $CATKIN_WS

RUN wget https://github.com/RethinkRobotics/baxter/raw/master/baxter.sh
RUN chmod u+x baxter.sh

#simulator setup
WORKDIR $CATKIN_WS/src
COPY baxter_simulator.rosinstall ./
RUN wstool merge baxter_simulator.rosinstall -y && \
    wstool update
RUN /bin/bash -c 'source /opt/ros/melodic/setup.bash; cd ..; catkin_make'

# melodic moveit setup
WORKDIR $CATKIN_WS/src
RUN git clone https://github.com/ros-planning/moveit_robots.git \
	&& apt-get update \
	&& apt-get install ros-melodic-moveit -y
# cd ~/baxter_ws/ | ./baxter.sh | catkin_make

# # # # # # # # # # # # # #
# tini subreap for VNC? #
# # # # # # # # # # # #

ENV TINI_VERSION v0.9.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini

ADD image /
RUN pip install setuptools wheel && pip install -r /usr/lib/web/requirements.txt

RUN cp /usr/share/applications/terminator.desktop /root/Desktop
RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc

EXPOSE 80
WORKDIR /root
ENV HOME=/home/ubuntu \
    SHELL=/bin/bash
ENTRYPOINT ["/startup.sh"]