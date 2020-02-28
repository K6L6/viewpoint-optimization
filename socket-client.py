#!/usr/bin/env python2
import struct
import cv2
import numpy as np
import socket
import json
import sys
import ipdb

import rospy
import rospkg
import baxter_interface
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
    return json.dumps(data)

def decode(data):
    return json.loads(data)

def gazeboPose2GQN_VP(camera_pos,offset):
    
    x, y, z = camera_pos - offset

    norm = np.linalg.norm([x,y,z]) #radius?
    yaw, pitch = compute_yaw_and_pitch([x,y,z])
    
    GQN_viewpoint = [x, y, z, np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)]
    
    return GQN_viewpoint

#class instance & init node
rospy.init_node('test',anonymous=True)
limb = 'left'
Bax = MoveBaxter(limb)
im = GetImage()

HOST = '192.168.170.209'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

## explicit starting position
starting_joint_angles = {'left_w0': 0.6699952259595108,
                            'left_w1': 1.030009435085784,
                            'left_w2': -0.4999997247485215,
                            'left_e0': -1.189968899785275,
                            'left_e1': 1.9400238130755056,
                            'left_s0': -0.08000397926829805,
                            'left_s1': -0.9999781166910306}
## obtain initial position
Bax._guarded_move_to_joint_position(starting_joint_angles)
init_pose = Pose()
init_pose.position.x, init_pose.position.y, init_pose.position.z, init_pose.orientation.x, init_pose.orientation.y, init_pose.orientation.z, init_pose.orientation.w = Bax.get_current_pose()

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
s.connect((HOST, PORT))
data = encode([observed_viewpoint.tolist(), observed_image.tolist(), offset.tolist()])

s.sendall(data)
s.getsockopt(socket.SOL_SOCKET,socket.SO_KEEPALIVE)
#data_recv=s.recv(1024)
#print('received',repr(data))
# ,observed_viewpoint)
# data = s.recv()

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
