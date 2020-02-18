#!/usr/bin/env python2
import os
import sys
import math
import argparse

import numpy as np
import cupy as cp
import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
import chainer
#import chainer.functions as cf
import sys
from chainer.backends import cuda

import gqn
from gqn.preprocessing import make_uint8,preprocess_images
from model_chain2 import Model
from hyperparams import HyperParameters
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

# WIP in Baxter Simulator
# obtain image and viewpoint from end-effector

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot-directory", type=str, required=True)
parser.add_argument("--snapshot-file", type=str, required=True)
parser.add_argument("--gpu-device", type=int, default=0)
args = parser.parse_args()

def compute_camera_angle_at_frame(t,total_frames):
    print("at frame: "+str(t)+" total frames: "+str(total_frames))
    return t*2*np.pi/total_frames

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

def rotate_query_viewpoint(horizontal_angle_rad, camera_distance,
                               camera_position_y):
    camera_position = np.array([
        camera_distance * math.sin(horizontal_angle_rad),  # x
        camera_position_y,
        camera_distance * math.cos(horizontal_angle_rad),
    ])
    center = np.array((0, 0, camera_position_z))
    camera_direction = camera_position - center
    yaw, pitch = compute_yaw_and_pitch(camera_direction)
    query_viewpoints = xp.array(
        (
            camera_position[0],
            camera_position[1],
            camera_position[2],
            np.sin(yaw),
            np.cos(yaw),
            np.sin(pitch),
            np.cos(pitch),
        ),
        dtype=np.float32,
    )
    query_viewpoints = xp.broadcast_to(query_viewpoints,
                                        (1, ) + query_viewpoints.shape)
    return query_viewpoints

def gazeboPose2GQN_VP(camera_pos,offset):
    
    x, y, z = camera_pos - offset

    norm = np.linalg.norm([x,y,z]) #radius?
    yaw, pitch = compute_yaw_and_pitch([x,y,z])
    
    GQN_viewpoint = [x, y, z, np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)]
    
    return GQN_viewpoint

def GQN_VP2gazeboPose(gqn_pos,offset):
    x, y, z = gqn_pos + offset

    gazebo_pose = x, y, z
    return gazebo_pose

if __name__ == "__main__":
    #class instance & init node
    rospy.init_node('test',anonymous=True)
    limb = 'left'
    Bax = MoveBaxter(limb)
    im = GetImage()
    
    # load model
    my_gpu = args.gpu_device
    if my_gpu < 0:
         xp=np
    else:
        cuda.get_device(args.gpu_device).use()
        xp=cp
    hyperparams = HyperParameters()
    assert hyperparams.load(args.snapshot_directory)

    model = Model(hyperparams)
    chainer.serializers.load_hdf5(args.snapshot_file, model)
    if my_gpu > -1:
        model.to_gpu()
    
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
    #ipdb.set_trace()
    observed_viewpoint = np.asarray(gazeboPose2GQN_VP([move.position.x, move.position.y, move.position.z],offset),dtype=np.float32)
    observed_viewpoint = np.expand_dims(np.expand_dims(observed_viewpoint, axis=0), axis=0)
    
    #ipdb.set_trace()
    camera_distance = np.mean(np.linalg.norm(observed_viewpoint[:,:,:3],axis=2))
    camera_position_z = np.mean(observed_viewpoint[:,:,1])
    
    # get current observed image from simulator
    observed_image = np.expand_dims(np.expand_dims(np.asarray(im.get()),axis=0),axis=0)
    observed_image = observed_image.transpose((0,1,4,2,3)).astype(np.float32)
    observed_image = preprocess_images(observed_image)

    # create representation and generate uncertainty map of environment [1000 viewpoints?]
    total_frames = 100
    
    representation = model.compute_observation_representation(observed_image, observed_viewpoint)

    # get predictions
    highest_var = 0.0
    no_of_samples = 100

    for i in range(0,total_frames):
        horizontal_angle_rad = compute_camera_angle_at_frame(i, total_frames)
        
        query_viewpoints = rotate_query_viewpoint(
                    horizontal_angle_rad, camera_distance, camera_position_z)
        #ipdb.set_trace()
        generated_images = xp.squeeze(xp.array(model.generate_images(query_viewpoints,
                                                            representation,no_of_samples)))
        var_image = xp.var(generated_images,axis=0)
        var_image = chainer.backends.cuda.to_cpu(var_image)
        # grayscale
        r,g,b = var_image
        gray_var_image = 0.2989*r+0.5870*g+0.1140*b
        current_var = np.mean(gray_var_image)
        
        if current_var>highest_var:
            highest_var = current_var
            highest_var_vp = query_viewpoints[0]
    
    ipdb.set_trace()
    # return next viewpoint and unit vector of end effector based on highest uncertainty found in the uncertainty map
    _x, _y, _z, _, _, _, _ = highest_var_vp
    _yaw, _pitch = compute_yaw_and_pitch([_x, _y, _z])
    pose_x, pose_y, pose_z = GQN_VP2gazeboPose([_x,_y,_z],offset) # convert _x, _y, _z to end effector values??
    quaternion_yaw = [0,0,np.sin(yaw/2),np.cos(yaw/2)]
    quaternion_pitch = [0,np.sin(pitch/2),0,np.cos(pitch/2)]
    move.position.x = pose_x
    move.position.y = pose_y
    move.position.z = pose_z
    quaternion_1 = tf.transformations.quaternion_multiply(quaternion_0, quaternion_yaw)
    quaternion_1 = tf.transformations.quaternion_multiply(quaternion_1, quaternion_pitch)
    move.orientation.w, move.orientation.x, move.orientation.y, move.orientation.z = quaternion_1

    joint_angles = Bax.get_joint_angles(move)
    Bax._guarded_move_to_joint_position(joint_angles)
    
    print("variance:"+str(highest_var))
    print("viewpoint data: "+str(highest_var_vp))
    print("Baxter end-effector coordinates:\n")
    print(move)

    # capture images as end effector is moving to next viewpoint [every 1 second?]

    # add image to representation vector and recalculate the representation map

    ## change direction of viewpoint mid-way if there is a higher uncertainty detected (how?)

    ## create movement for task arm and generate a unit vector to face the grasping point of object
