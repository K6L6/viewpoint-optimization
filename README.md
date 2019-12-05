# Viewpoint Optimization for Robot Tasks through Uncertainty Represented in World Models    :construction_worker:
The purpose is to compensate for limitations of performing robot tasks with a fixed single viewpoint.  
This is a method developed to learn optimized viewpoint for robots to perform the designated task.  
Next suitable viewpoint will be selected by the hill climbing algorithm, which will search for a viewpoint that shows highest uncertainty in the world model of the agent.

## Project Progress
* [x] Prelimenary experiments
    1. [x] Observation of variance in predictions
    2. [x] Observation of variance in representation vector
    3. [x] Observation of mean squared error of predictions
* [x] Setup for data collection
    1. [x] Docker container for Gazebo simulator
    2. [x] Motion generation with Baxter simulator
    3. [x] Data collection of images and viewpoints of 3D objects
    4. [x] Without domain randomization
    5. [ ] Apply domain randomization :snail:
* [ ] Train GQN to recognize uncertainty from simulator dataset (in progress)
    1. [x] Reconstruct GQN code to support Chainer API [iterators, trainer, extensions, etc.]
    2. [x] Reconstruct GQN code to support single node multi-GPU training (in progress)
    3. [ ] Obtain reasonable predictions from trained model, to begin experiments
* [ ] Apply trained GQN to move Baxter arm for viewpoint optimization
* [ ] Train the neural network to move task arm regardless of observed viewpoint
* [ ] Perform grasping task
* [ ] Perform other robot tasks (if possible)


## Required Packages
nvidia-docker: https://github.com/NVIDIA/nvidia-docker
```
# Add the package repositories
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```
Chainer implementation of GQN: https://github.com/musyoku/chainer-gqn  
```
git clone https://github.com/musyoku/chainer-gqn.git
```
### GQN Docker Usage
build the GQN Docker image:  
```
docker build -t [docker-image name] .
```  
  
create the container from GQN docker image:  
```
nvidia-docker run -it --name [name of container] -v [local/directory/of/gqn/repo]:[container/directory/of/gqn/repo] -v [local/directory/of/dataset]:[container/directory/of/dataset] [docker-image name]
```  
  
conda environment needs to be activated to access libraries  
```. /miniconda/bin/activate```
  
### Baxter Simulator Docker Usage
This Docker image uses a VNC method to provide a display for the Gazebo simulator while running the container.  
This repository was used as a reference to build this image: https://github.com/fcwu/docker-ubuntu-vnc-desktop  
as well as the Rethink Robotics webpage for Baxter simulator installation: https://github.com/RethinkRobotics/sdk-docs/wiki/Baxter-simulator  
**Specifications:**
* Ubuntu 16.04
* ROS Kinetic
* Gazebo 7.15
* includes RViz
  
First, build the Baxter Simulator Docker image (building may take a while because CUDA supported Ubuntu image will also be built):  
```
docker build -t [docker-image name] .
```  
  
create the container from the image:  
```
nvidia-docker run -it -p 6080:80 [docker-image name]
```  
  
after running the container, these messages should show up on the terminal:  

```sh
/usr/lib/python2.7/dist-packages/supervisor/options.py:297: UserWarning: Supervisord is running as root and it is searching for its configuration file in default locations (including its current working directory); you probably want to specify a "-c" argument specifying an absolute path to a configuration file for improved security.
  'Supervisord is running as root and it is searching '
2019-11-15 06:12:41,054 CRIT Supervisor running as root (no user in config file)
2019-11-15 06:12:41,054 WARN Included extra file "/etc/supervisor/conf.d/supervisord.conf" during parsing
2019-11-15 06:12:41,069 INFO RPC interface 'supervisor' initialized
2019-11-15 06:12:41,070 CRIT Server 'unix_http_server' running without any HTTP authentication checking
2019-11-15 06:12:41,070 INFO supervisord started with pid 22
2019-11-15 06:12:42,072 INFO spawned: 'xvfb' with pid 26
2019-11-15 06:12:42,074 INFO spawned: 'pcmanfm' with pid 27
2019-11-15 06:12:42,075 INFO spawned: 'lxpanel' with pid 28
2019-11-15 06:12:42,077 INFO spawned: 'lxsession' with pid 29
2019-11-15 06:12:42,079 INFO spawned: 'x11vnc' with pid 30
2019-11-15 06:12:42,081 INFO spawned: 'novnc' with pid 31
2019-11-15 06:12:42,678 INFO exited: lxsession (exit status 1; not expected)
2019-11-15 06:12:43,162 INFO success: xvfb entered RUNNING state, process has stayed up for > than 1 seconds (startsecs)
2019-11-15 06:12:43,162 INFO success: pcmanfm entered RUNNING state, process has stayed up for > than 1 seconds (startsecs)
2019-11-15 06:12:43,162 INFO success: lxpanel entered RUNNING state, process has stayed up for > than 1 seconds (startsecs)
2019-11-15 06:12:43,162 INFO success: x11vnc entered RUNNING state, process has stayed up for > than 1 seconds (startsecs)
2019-11-15 06:12:43,162 INFO success: novnc entered RUNNING state, process has stayed up for > than 1 seconds (startsecs)
2019-11-15 06:12:44,430 INFO spawned: 'lxsession' with pid 43
2019-11-15 06:12:45,432 INFO success: lxsession entered RUNNING state, process has stayed up for > than 1 seconds (startsecs)
```

then open your browser and enter the URL in the address bar:  
```http://127.0.0.1:6080```  
  
the browser should load the desktop below:  
[![link](/result_example/baxter_sim_vncdocker.png)](gitlab.com/K6L6/hidden-perspective-discovery/blob/master/)   
## Dataset details
* created in Gazebo
* uses object models from 3DGEMS
* 128 scenes (selected by removing overly large objects)
* 19 images and viewpoints per scene

## Some training results

Simulator dataset with Baxter and 3DGEMS dataset (size will be increased by domain randomization) :soon:  
### Simulator Dataset ###
Training with 0.002% of GQN's rooms_free_camera dataset for 100 epochs takes approx 1 day with 1 GPU.  
Training with dataset synthesized from simulator takes approx a little more than 3 hours for 1000 epochs.  
Image size seems to be directly proportional to training time.  
  
Rendering of a **thin green book**.  
[![link](/result_example/green_book.gif)](gitlab.com/K6L6/hidden-perspective-discovery/blob/master/)  
  
Rendering of a **blue pen**.  
[![link](/result_example/blue_pen.gif)](gitlab.com/K6L6/hidden-perspective-discovery/blob/master/)  
  
Some thoughts on training result:  
* Recently found out that GQN uses Y axis for height, which differed from Baxter simulator that used Z axis for height.

### Rooms Free Camera without Object Rotations ###
finished training but result not yet checked.
