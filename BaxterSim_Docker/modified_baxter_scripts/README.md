on baxter.sh
change bax_ID: 011604P0017
change IP: 172.17.0.2
change version: kinetic

/cam_attachment goes in rethink_ee_description/urdf/

left_end_effectormod.urdf.xacro & baxter.urdf.xacro & baxter_base.gazebo.xacro can be found in /baxter_description

baxter_world.launch is a launch file in /src/baxter_simulator/baxter_gazebo/launch

paths need to be changed/added for  'gazebo_model_path' in the package.xml of baxter_sim_examples, so that objects spawned are visible in the simulator.
add ":/home/baxter_ws/models/decoration:/home/baxter_ws/models/earthquake:/home/baxter_ws/models/electronics:/home/baxter_ws/models/food:/home/baxter_ws/models/kitchen:/home/baxter_ws/models/shapes:/home/baxter_ws/models/stationery" 

in /home/baxter_ws: do rsync -rz kelvin@192.xxx.xxx.xxx:/home/kelvin/OgataLab/hidden-perspective-discovery/BaxterSim_Docker/simulator_dataset/models ./

Image size taken by camera can be changed in baxter_base.gazebo.xacro under "cam_link"
