#!/usr/bin/env python2
# finds transformation matrix

import numpy as np
from simulator_observer import MoveBaxter
import ipdb

import rospy
import rospkg
import baxter_interface
import tf

from geometry_msgs.msg import (
    PoseStamped, 
    Pose, 
    Point, 
    Quaternion,
)

rospy.init_node("matrix_trans",anonymous=True)
# matrix x
rad = 0.2
theta = np.linspace(0,np.pi*2,10)
circle_x = -rad*np.cos(theta)
circle_y = rad*np.sin(theta)

matrix_x = np.zeros((10,3)) # gqn coordinates
matrix_x[:,0] = circle_x 
matrix_x[:,1] = circle_y

# shift for baxter movement
theta_shift = np.zeros(len(theta)-1)
circle_shift_x = np.zeros(len(circle_x)-1)
circle_shift_y = np.zeros(len(circle_y)-1)

for i in range(len(circle_shift_x)):
    circle_shift_x[i] = circle_x[i+1]-circle_x[i]
    circle_shift_y[i] = circle_y[i+1]-circle_y[i]
    theta_shift[i] = theta[i+1]-theta[i]

# use tf for transformer


# move baxter to obtain end effector coordinates
Bax = MoveBaxter('left')

## move to initial position
starting_joint_angles = {'left_w0': 0.6699952259595108,
                             'left_w1': 1.030009435085784,
                             'left_w2': -0.4999997247485215,
                             'left_e0': -1.189968899785275,
                             'left_e1': 1.9400238130755056,
                             'left_s0': -0.08000397926829805,
                             'left_s1': -0.9999781166910306}

Bax._guarded_move_to_joint_position(starting_joint_angles)

move = Pose()
move.position.x, move.position.y, move.position.z, move.orientation.x, move.orientation.y, move.orientation.z, move.orientation.w = Bax.get_current_pose()

origin_x_in_y = np.asarray([move.position.x,move.position.y,move.position.z])
move.position.x -= 0.2
next_joint_angles = Bax.get_joint_angles(move)
Bax._guarded_move_to_joint_position(next_joint_angles)

move.position.z -= 0.37
next_joint_angles = Bax.get_joint_angles(move)
Bax._guarded_move_to_joint_position(next_joint_angles)

matrix_y = np.zeros((10,3)) # baxter end effector coordinates
matrix_y[0] = np.asarray([move.position.x, move.position.y, move.position.z])

for i in range(len(circle_shift_x)):
    move.position.x += circle_shift_x[i]
    move.position.y += circle_shift_y[i]
    matrix_y[i+1] = np.asarray([move.position.x, move.position.y, move.position.z])
    next_joint_angles = Bax.get_joint_angles(move)
    Bax._guarded_move_to_joint_position(next_joint_angles)

# calculate transformation matrix i.e. transform y to x
A = np.vstack([matrix_y.T, np.ones(10)]).T

# perform the least squares method
x, res, rank, s = np.linalg.lstsq(A, matrix_x, rcond=None)

ipdb.set_trace()
# test results
print(np.dot(A,x))