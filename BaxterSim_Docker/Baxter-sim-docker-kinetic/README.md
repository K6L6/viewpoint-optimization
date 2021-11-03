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
