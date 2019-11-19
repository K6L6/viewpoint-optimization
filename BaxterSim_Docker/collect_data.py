#!/usr/bin/env python
# obtain rostopic information with python

import argparse
import struct
import sys
import os
import copy
import glob
import ipdb
import numpy as np
import tf

from save_image import SaveImage

import rospy
import rospkg
import baxter_interface

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

class RotateArm(object):
    def __init__(self, limb, hover_distance=0.15, verbose=True):
        self._limb_name = limb # string
        self._hover_distance = hover_distance # in meters
        self._verbose = verbose # bool
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
    
    def get_joint_angles(self,pose):
        # pose = self.get_current_position() # obtain current pose cartesian[x,y,z] and orientation[x,y,z,w]
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq =  SolvePositionIKRequest() # Baxter IK Solver
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            joint_angles =  self._iksvc(ikreq) # request joint angles
        except (rospy.ServiceException, rospy.ROSException), e:
            ROSPY.LOGERR("Service call failed: %s" % (e,))
            return False
        
        resp_seeds = struct.unpack('<%dB' % len(joint_angles.result_type),joint_angles.result_type)
        limb_joints = {}
        if (resp_seeds[0] != joint_angles.RESULT_INVALID):
            seed_str={
                ikreq.SEED_USER: 'User Provided Seed',
                ikreq.SEED_CURRENT: 'Current Joint Angles',
                ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
            }.get(resp_seeds[0],'None')

            if self._verbose:
                # print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type:{0}".format((seed_str)))
                limb_joints = dict(zip(joint_angles.joints[0].name, joint_angles.joints[0].position))
        else:
            rospy.logerr("INVALID POSE")
            return False
        
        return limb_joints
    
    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("no joint angles provided")
    
    def calc_shift_y(self,x,r):
        shift = np.zeros(len(x))
        for i in range(len(x)):
            shift[i] = np.sqrt(r**2-x[i]**2)
        
        for i in range(len(x)):
            if (i+1)==len(x):
                break
            else:
                shift[i] = shift[i+1]-shift[i]
        
        return shift
    
    def deg_to_rad(self,degree):
        rad = degree/180.0*np.pi    
        return rad

    def get_viewpoint(self):
        pose = self._limb.endpoint_pose()
        # get [x,y,z,sin(yaw),cos(yaw),sin(pitch),cos(pitch)]
        r, p, y = tf.transformations.euler_from_quaternion([pose['orientation'].w,pose['orientation'].x,pose['orientation'].y,pose['orientation'].z],axes='sxyz')
        viewpoint = np.array([pose['position'].x,pose['position'].y,pose['position'].z,np.sin(y),np.cos(y),np.sin(p),np.cos(p)])

        return viewpoint

    def rotate_arm(self, radius, steps, object_name):
        starting_joint_angles = {'left_w0': 0.6699952259595108,
                             'left_w1': 1.030009435085784,
                             'left_w2': -0.4999997247485215,
                             'left_e0': -1.189968899785275,
                             'left_e1': 1.9400238130755056,
                             'left_s0': -0.08000397926829805,
                             'left_s1': -0.9999781166910306}

        self._guarded_move_to_joint_position(starting_joint_angles)

        current_pose = self._limb.endpoint_pose()
        rotate = Pose()
        rotate.position.x = current_pose['position'].x
        rotate.position.y = current_pose['position'].y
        rotate.position.z = current_pose['position'].z
        rotate.orientation.x = current_pose['orientation'].x # twist of grippers
        rotate.orientation.y = current_pose['orientation'].y 
        rotate.orientation.z = current_pose['orientation'].z 
        rotate.orientation.w = current_pose['orientation'].w 
    
        ## Rotate
        step_value = radius/steps
        circle_x = np.arange(-radius,radius+step_value, step=step_value)
        move_steps = len(circle_x)
        circle_yp = np.zeros(move_steps)
        circle_shift_yp = np.zeros(move_steps)
        circle_shift_x = np.zeros(move_steps)
        wrist_twist = 3.059/move_steps # value obtained from baxter hardware specifications for w2
        for i in range(move_steps):
            if (i+1)==move_steps:
                break
            else:    
                circle_shift_x[i] = circle_x[i+1] - circle_x[i]    
        circle_shift_yp = self.calc_shift_y(circle_x,radius)
        rotate.orientation.x=0
        # self.get_data(rotate)

        ## Save params ##
        object_name = object_name
        directory = "/home/baxter_ws/images/"+object_name+"/"
        if os.path.exists(directory):
            pass
        else:
            os.makedirs(directory)
        
        file_number = 0
        
        ##############################
        ### Algorithm for rotation ###
        ##############################
        rotate.position.z -= 0.3
        rotate.position.x -= radius
    
        joint_angles = self.get_joint_angles(rotate)
        self._guarded_move_to_joint_position(joint_angles)
        print("bottom layer start ")
        z_rotation_step = (2*np.pi)/(4*steps)    
        
        quaternion_00 = [rotate.orientation.x, rotate.orientation.y, rotate.orientation.z, rotate.orientation.w]
        quaternion_res = quaternion_00

        viewpoints = np.empty((0,7),dtype=np.float32)

        # # # # # # # # # # # # # # # # # # # # # # # # #
        ### ### ### ### Lowest layer of Dome ### ### ###                                #### BOTTOM ####
        print("Rotating bottom layer...",rotate.orientation)

        print("rotate first half")
        im = SaveImage(directory)
        im.file_number = file_number
        last_saved_file_number = im.last_saved_file_number  # saving algorithm
        im.snap_flag = True
        while(True):
            if im.last_saved_file_number == file_number:
                if im.snap_flag == False:
                    file_number += 1
                    break
        viewpoints = np.append(viewpoints,[self.get_viewpoint()], axis=0)

        for j in range(move_steps-1):
            rotate.position.x += circle_shift_x[j]
            rotate.position.y += circle_shift_yp[j]
            
            quaternion_z = [0,0,np.sin(z_rotation_step/2),np.cos(z_rotation_step/2)]
            quaternion_res = tf.transformations.quaternion_multiply(quaternion_res,quaternion_z)
            rotate.orientation.w = quaternion_res[3]
            rotate.orientation.x = quaternion_res[0]
            rotate.orientation.y = quaternion_res[1]
            rotate.orientation.z = quaternion_res[2]
            
            joint_angles = self.get_joint_angles(rotate)
            self._guarded_move_to_joint_position(joint_angles)
            
            im.file_number = file_number
            last_saved_file_number = im.last_saved_file_number
            im.snap_flag = True
            while(True):
                if im.last_saved_file_number == file_number:
                    if im.snap_flag == False:
                        file_number += 1
                        break
            viewpoints = np.append(viewpoints,[self.get_viewpoint()], axis=0)
        # ipdb.set_trace()
        print("rotate second half")
        
        for j in range(move_steps-1):
            rotate.position.x -= circle_shift_x[j]
            rotate.position.y -= circle_shift_yp[j]

            quaternion_z = [0,0,np.sin(z_rotation_step/2),np.cos(z_rotation_step/2)]
            quaternion_res = tf.transformations.quaternion_multiply(quaternion_res,quaternion_z)
            rotate.orientation.w = quaternion_res[3]
            rotate.orientation.x = quaternion_res[0]
            rotate.orientation.y = quaternion_res[1]
            rotate.orientation.z = quaternion_res[2]
            
            joint_angles = self.get_joint_angles(rotate)
            self._guarded_move_to_joint_position(joint_angles)

            im.file_number = file_number
            last_saved_file_number = im.last_saved_file_number
            im.snap_flag = True
            while(True):
                if im.last_saved_file_number == file_number:
                    if im.snap_flag == False:
                        file_number += 1
                        break
            viewpoints = np.append(viewpoints,[self.get_viewpoint()], axis=0)

        # # # # # # # # # # # # # # # # # # # # # # # # #
        ### ### ### ### Middle layer of Dome ### ### ###                                #### MIDDLE ####
        print("centering")
        self._guarded_move_to_joint_position(starting_joint_angles)
        
        print("elevate")
        height = radius*0.5
        new_rad = np.sqrt(radius**2-height**2)
        step_value = new_rad/steps
        circle_x = np.arange(-new_rad,new_rad+step_value, step=step_value)
        for i in range(move_steps):
            if (i+1)==move_steps:
                break
            else:    
                circle_shift_x[i] = circle_x[i+1] - circle_x[i]
        circle_shift_yp = self.calc_shift_y(circle_x,new_rad)
        rotate.position.z += height # elevate
        rotate.position.x += (radius-new_rad)
        joint_angles = self.get_joint_angles(rotate)
        self._guarded_move_to_joint_position(joint_angles)
        
        print("mid layer start ")
        theta = np.pi/6 # elevation angle

        rotate.orientation.w = quaternion_00[3]
        rotate.orientation.x = quaternion_00[0]
        rotate.orientation.y = quaternion_00[1]
        rotate.orientation.z = quaternion_00[2]
        joint_angles = self.get_joint_angles(rotate)
        self._guarded_move_to_joint_position(joint_angles)

        quaternion_y = [0,np.sin(theta/2),0,np.cos(theta/2)]
        quaternion_res = tf.transformations.quaternion_multiply(quaternion_00,quaternion_y)
        rotate.orientation.w = quaternion_res[3]
        rotate.orientation.x = quaternion_res[0]
        rotate.orientation.y = quaternion_res[1]
        rotate.orientation.z = quaternion_res[2]
        joint_angles = self.get_joint_angles(rotate)
        self._guarded_move_to_joint_position(joint_angles)

        im.file_number = file_number
        last_saved_file_number = im.last_saved_file_number
        im.snap_flag = True
        while(True):
            if im.last_saved_file_number == file_number:
                if im.snap_flag == False:
                    file_number += 1
                    break
        viewpoints = np.append(viewpoints,[self.get_viewpoint()], axis=0)

        print("rotate upper first half",rotate.orientation)

        for j in range(move_steps-1):
            rotate.position.x += circle_shift_x[j]
            rotate.position.y += circle_shift_yp[j]
            quaternion_z = [0,0,np.sin((z_rotation_step*(j+1))/2),np.cos(z_rotation_step*(j+1)/2)]
            quaternion_res = tf.transformations.quaternion_multiply(quaternion_00,quaternion_z)
            quaternion_tmp = quaternion_res
            quaternion_res = tf.transformations.quaternion_multiply(quaternion_res,quaternion_y)
            rotate.orientation.w = quaternion_res[3]
            rotate.orientation.x = quaternion_res[0]
            rotate.orientation.y = quaternion_res[1]
            rotate.orientation.z = quaternion_res[2]
            if j%2==0:
                joint_angles = self.get_joint_angles(rotate)
                self._guarded_move_to_joint_position(joint_angles)

                im.file_number = file_number
                last_saved_file_number = im.last_saved_file_number
                im.snap_flag = True
                while(True):
                    if im.last_saved_file_number == file_number:
                        if im.snap_flag == False:
                            file_number += 1
                            break
                viewpoints = np.append(viewpoints,[self.get_viewpoint()], axis=0)
        
        print("rotate upper second half")
        for j in range(move_steps-1):
            rotate.position.x -= circle_shift_x[j]
            rotate.position.y -= circle_shift_yp[j]
            quaternion_z = [0,0,np.sin((z_rotation_step*(move_steps+j))/2),np.cos((z_rotation_step*(move_steps+j))/2)]
            quaternion_res = tf.transformations.quaternion_multiply(quaternion_00,quaternion_z)
            quaternion_tmp = quaternion_res
            quaternion_res = tf.transformations.quaternion_multiply(quaternion_res,quaternion_y)
            rotate.orientation.w = quaternion_res[3]
            rotate.orientation.x = quaternion_res[0]
            rotate.orientation.y = quaternion_res[1]
            rotate.orientation.z = quaternion_res[2]
            if j%2==0:
                joint_angles = self.get_joint_angles(rotate)
                self._guarded_move_to_joint_position(joint_angles)
                im.file_number = file_number
                last_saved_file_number = im.last_saved_file_number
                im.snap_flag = True
                while(True):
                    if im.last_saved_file_number == file_number:
                        if im.snap_flag == False:
                            file_number += 1
                            break
                viewpoints = np.append(viewpoints,[self.get_viewpoint()], axis=0)
        
        # # # # # # # # # # # # # # # # # # # # # # # # #
        ### ### ### ### Top layer of Dome ### ### ###                                #### TOP ####
        print("centering")
        self._guarded_move_to_joint_position(starting_joint_angles)
        
        print("elevate")
        height = radius*0.75
        new_rad = np.sqrt(radius**2-height**2)
        step_value = new_rad/steps
        circle_x = np.arange(-new_rad,new_rad+step_value, step=step_value)
        for i in range(move_steps):
            if (i+1)==move_steps:
                break
            else:    
                circle_shift_x[i] = circle_x[i+1] - circle_x[i]
        circle_shift_yp = self.calc_shift_y(circle_x,new_rad)
        rotate.position.z += radius*0.25 # elevate
        rotate.position.x += (radius-new_rad)
        joint_angles = self.get_joint_angles(rotate)
        self._guarded_move_to_joint_position(joint_angles)
        
        print("top layer start ")
        theta = np.arcsin(3.0/4.0) # elevation angle

        rotate.orientation.w = quaternion_00[3]
        rotate.orientation.x = quaternion_00[0]
        rotate.orientation.y = quaternion_00[1]
        rotate.orientation.z = quaternion_00[2]
        joint_angles = self.get_joint_angles(rotate)
        self._guarded_move_to_joint_position(joint_angles)
 
        quaternion_y = [0,np.sin(theta/2),0,np.cos(theta/2)]
        quaternion_res = tf.transformations.quaternion_multiply(quaternion_00,quaternion_y)
       
        rotate.orientation.w = quaternion_res[3]
        rotate.orientation.x = quaternion_res[0]
        rotate.orientation.y = quaternion_res[1]
        rotate.orientation.z = quaternion_res[2]
        joint_angles = self.get_joint_angles(rotate)
        self._guarded_move_to_joint_position(joint_angles)

        im.file_number = file_number
        last_saved_file_number = im.last_saved_file_number
        im.snap_flag = True
        while(True):
            if im.last_saved_file_number == file_number:
                if im.snap_flag == False:
                    file_number += 1
                    break
        viewpoints = np.append(viewpoints,[self.get_viewpoint()], axis=0)

        print("top layer first half",rotate.orientation)

        for j in range(move_steps-1):
            rotate.position.x += circle_shift_x[j]
            rotate.position.y += circle_shift_yp[j]
            quaternion_z = [0,0,np.sin((z_rotation_step*(j+1))/2),np.cos(z_rotation_step*(j+1)/2)]
            quaternion_res = tf.transformations.quaternion_multiply(quaternion_00,quaternion_z)
            quaternion_tmp = quaternion_res
            quaternion_res = tf.transformations.quaternion_multiply(quaternion_res,quaternion_y)
            rotate.orientation.w = quaternion_res[3]
            rotate.orientation.x = quaternion_res[0]
            rotate.orientation.y = quaternion_res[1]
            rotate.orientation.z = quaternion_res[2]
            if j%2==0:
                joint_angles = self.get_joint_angles(rotate)
                self._guarded_move_to_joint_position(joint_angles)
                im.file_number = file_number
                last_saved_file_number = im.last_saved_file_number
                im.snap_flag = True
                while(True):
                    if im.last_saved_file_number == file_number:
                        if im.snap_flag == False:
                            file_number += 1
                            break
                viewpoints = np.append(viewpoints,[self.get_viewpoint()], axis=0)

        print("top layer second half")
        for j in range(move_steps-1):
            rotate.position.x -= circle_shift_x[j]
            rotate.position.y -= circle_shift_yp[j]
            quaternion_z = [0,0,np.sin((z_rotation_step*(move_steps+j))/2),np.cos((z_rotation_step*(move_steps+j))/2)]
            quaternion_res = tf.transformations.quaternion_multiply(quaternion_00,quaternion_z)
            quaternion_tmp = quaternion_res
            quaternion_res = tf.transformations.quaternion_multiply(quaternion_res,quaternion_y)
            rotate.orientation.w = quaternion_res[3]
            rotate.orientation.x = quaternion_res[0]
            rotate.orientation.y = quaternion_res[1]
            rotate.orientation.z = quaternion_res[2]
            if j%2==0:
                joint_angles = self.get_joint_angles(rotate)
                self._guarded_move_to_joint_position(joint_angles)
                im.file_number = file_number
                last_saved_file_number = im.last_saved_file_number
                im.snap_flag = True
                while(True):
                    if im.last_saved_file_number == file_number:
                        if im.snap_flag == False:
                            file_number += 1
                            break
                viewpoints = np.append(viewpoints,[self.get_viewpoint()], axis=0)
        
        np.save(directory+"viewpoints.npy",viewpoints)

def load_gazebo_models(table_pose=Pose(position=Point(x=0.8, y=0.0, z=0.0)),
                       table_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('baxter_sim_examples')+"/models/"
    # Load Table SDF
    table_xml = ''
    with open (model_path + "cafe_table/model.sdf", "r") as table_file:
        table_xml=table_file.read().replace('\n', '')
    
    # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("cafe_table", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
      

def main():
    # rospy.wait_for_message("robot/sim/started",Empty)
    rospy.init_node("rotate_baxter_arm")
    # print("boo")
    print("loading models...")
    load_gazebo_models()
    limb = 'left'
    hover_distance = 0.15 # meters
    arm = RotateArm(limb, hover_distance)
    object_pose=Pose(position=Point(x=0.6, y=0.2, z=0.8))
    # object_pose = Pose(position=Point(x=0.6725, y=0.1265, z=0.7825)) # from baxter sim ik demo
    object_reference_frame="world"
    # Spawn model sdf
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    rospy.wait_for_service('/gazebo/delete_model')

    database_directory = "/home/baxter_ws/models/"
    for folder_path in glob.glob(database_directory+'**/'):
        print(folder_path)
        object_xml = ''
        for path in glob.glob(folder_path+'**/'):
            print(path)
            object_name = path.split('/')[-2]
            print(object_name)
            with open (path+"model.sdf","r") as object_file:
                object_xml=object_file.read().replace('\n','')
            
            try:
                spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
                resp_sdf = spawn_sdf(object_name, object_xml, "",
                                    object_pose, object_reference_frame)
            except rospy.ServiceException, e:
                rospy.logerr("Spawn SDF service call failed: {0}".format(e))
        
            # while not rospy.is_shutdown():
            print("\nRotating arm...")
            arm.rotate_arm(0.2,2,object_name)

            try:
                delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
                resp_delete = delete_model(object_name)
            except rospy.ServiceException, e:
                rospy.loginfo("Delete Model service call failed: {0}".format(e))
    
    resp_delete = delete_model('cafe_table')
    return 0

if __name__ == '__main__':
    sys.exit(main())
# obtain end-effector location from rostopic echo -n1 /robot/limb/left/endpoint_state
# obtain data image first, viewpoint next