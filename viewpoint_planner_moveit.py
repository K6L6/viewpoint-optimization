#!/usr/bin/env python2
import struct
import cv2
import numpy as np
import socket
import json
import copy
import math
import sys
from time import sleep
import cPickle as pickle
import ipdb
from six.moves import queue

import rospy
import rospkg
import baxter_interface
import moveit_commander
import moveit_msgs.msg
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from sensor_msgs.msg import Image
from simulator_observer import GetImage, MoveBaxter

from gazebo_msgs.srv import (
    SpawnModel, 
    DeleteModel,
)

from geometry_msgs.msg import (
    PoseStamped, 
    Pose, 
    Point, 
    Quaternion,
)

from sensor_msgs.msg import Image
from std_msgs.msg import (
    Header, 
    Empty,
)

from baxter_core_msgs.srv import (
    SolvePositionIK, 
    SolvePositionIKRequest,
)

def compute_yaw_and_pitch(vec):
    norm = np.linalg.norm(vec)
    x, y, z = vec
    
    if y<0:
        yaw = np.pi+np.arctan2(x,y)
    elif x<0:
        yaw = np.pi*2+np.arctan2(x,y)
    else:
        yaw = np.arctan2(x,y)
        
    pitch = -np.arcsin(z/norm)
    return yaw, pitch

def encode(data):
    return pickle.dumps(data,protocol=2)

def decode(data):
    return pickle.loads(data)

def gazeboPose2GQN_VP(camera_pos,offset):
    
    x, y, z = camera_pos - offset

    norm = np.linalg.norm([x,y,z]) #radius?
    yaw, pitch = compute_yaw_and_pitch([x,y,z])
    
    GQN_viewpoint = [x, y, z, np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)]
    
    return GQN_viewpoint

def GQN_VP2gazeboPose(camera_pos,offset):
    
    x, y, z = camera_pos + offset

    # norm = np.linalg.norm([x,y,z]) #radius?
    # yaw, pitch = compute_yaw_and_pitch([x,y,z])
    
    GQN_viewpoint = [x, y, z]
    
    return GQN_viewpoint

#class instance & init node & moveit_commander
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('test',anonymous=True)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
# limb = 'left'
# Bax = MoveBaxter(limb)
im = GetImage()

HOST = '192.168.170.209'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

group_name = "baxter_left_arm"
group = moveit_commander.MoveGroupCommander(group_name)

##trajectory publisher
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

## obtaining basic info
planning_frame = group.get_planning_frame()
print("= = = Reference Frame: %s" % planning_frame)
end-eff_link = group.get_end_effector_link()
print("= = = End Effector: %s" % end-eff_link)
group_names = robot.get_group_names()
print("= = = Robot Groups", robot.get_group_names())
print("= = = Robot State . . .")
print(robot.get_current_state())
print(" = = = = = = = = = = = = = ")

## explicit starting position
starting_joint_angles = {'left_w0': 0.6699952259595108,
                            'left_w1': 1.030009435085784,
                            'left_w2': -0.4999997247485215,
                            'left_e0': -1.189968899785275,
                            'left_e1': 1.9400238130755056,
                            'left_s0': -0.08000397926829805,
                            'left_s1': -0.9999781166910306}
## obtain initial position
joint_goal = group.get_current_joint_values()
print(joint_goal)
sys.exit()
# Bax._guarded_move_to_joint_position(starting_joint_angles)
# init_pose = Pose()
# init_pose.position.x, init_pose.position.y, init_pose.position.z, init_pose.orientation.x, init_pose.orientation.y, init_pose.orientation.z, init_pose.orientation.w = Bax.get_current_pose()

## object to store movement of end-effector
move = Pose()
move.position.x, move.position.y, move.position.z, move.orientation.x, move.orientation.y, move.orientation.z, move.orientation.w = Bax.get_current_pose()
move.position.x -= 0.2

next_joint_angles = Bax.get_joint_angles(move)
Bax._guarded_move_to_joint_position(next_joint_angles)
move.position.z -=0.37
next_joint_angles = Bax.get_joint_angles(move)
Bax._guarded_move_to_joint_position(next_joint_angles)

offset = np.asarray([move.position.x, move.position.y, move.position.z]) - np.asarray([-0.2,0.0,0.0])

observed_viewpoint = np.asarray(gazeboPose2GQN_VP([move.position.x, move.position.y, move.position.z],offset),dtype=np.float32)
#observed_viewpoint = np.expand_dims(np.expand_dims(observed_viewpoint, axis=0), axis=0)

#camera_distance = np.mean(np.linalg.norm(observed_viewpoint[:,:,:3],axis=2))
#camera_position_z = np.mean(observed_viewpoint[:,:,1])
camera_distance = np.mean(np.linalg.norm(observed_viewpoint[:3]))
camera_position_z = np.mean(observed_viewpoint[1])

# get current observed image from simulator
observed_image = np.asarray(im.get())
#observed_image = np.expand_dims(np.expand_dims(np.asarray(im.get()),axis=0),axis=0)

# send image and viewpoint to GQN

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
while True:    
    try:
        s.connect((HOST, PORT))
        break
    except socket.error:
        print('not alive')
        sleep(3)
        continue
    
data = [observed_viewpoint, observed_image, offset]
s.send(encode(data))

# s.getsockopt(socket.SOL_SOCKET,socket.SO_KEEPALIVE)
RECV_BUFFER = 131072
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#s.bind((HOST, PORT))
data_r = queue.Queue(maxsize=1)
while True:
    try:
        data = s.recv(RECV_BUFFER)
        data_r.put(decode(data))
        if not data: break
        
    except socket.error:
        s.close()
        print("communication cut")
        break
         

print(data_r.get())
next_vp = data_r.get()
x,y,z,yaw,pitch = next_vp
camera_pos = [x,y,z]
pose_x, pose_y, pose_z = GQN_VP2gazeboPose(camera_pos,offset)



# quaternion_yaw = [0,0,np.sin(yaw/2),np.cos(yaw/2)]
# quaternion_pitch = [0,np.sin(pitch/2),0,np.cos(pitch/2)]
# move.position.x = pose_x
# move.position.y = pose_y
# move.position.z = pose_z
# quaternion_1 = tf.transformations.quaternion_multiply(quaternion_0, quaternion_yaw)
# quaternion_1 = tf.transformations.quaternion_multiply(quaternion_1, quaternion_pitch)
# move.orientation.w, move.orientation.x, move.orientation.y, move.orientation.z = quaternion_1

# joint_angles = Bax.get_joint_angles(move)
# Bax._guarded_move_to_joint_position(joint_angles)

# print("variance:"+str(highest_var))
# print("viewpoint data: "+str(highest_var_vp))
# print("Baxter end-effector coordinates:\n")
# print(move)