#!/usr/bin/env python3
import os
import sys
import math
import argparse

import numpy as np
import cupy as cp
import socket
import cv2

import chainer
from chainer.backends import cuda

# import gqn
# from gqn.preprocessing import make_uint8, preprocess_images
# from model_chain2 import Model
# from hyperparams import HyperParameters
import ipdb

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

HOST = '0.0.0.0'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            conn.sendall(data)

ipdb.set_trace()
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


# receive image and viewpoint from baxter
#
#
#

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

# return next viewpoint and unit vector of end effector based on highest uncertainty found in the uncertainty map
_x, _y, _z, _, _, _, _ = highest_var_vp
_yaw, _pitch = compute_yaw_and_pitch([_x, _y, _z])
pose_x, pose_y, pose_z = GQN_VP2gazeboPose([_x,_y,_z],offset) # convert _x, _y, _z to end effector values??

#send x,y,z and yaw, pitch to baxter
#
#
#
