#!/usr/bin/env python
import rospy
import struct
import cv2
import matplotlib.pyplot as plt
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import tf
import ipdb

import baxter_interface
from geometry_msgs.msg import (
    PoseStamped, 
    Pose, 
    Point, 
    Quaternion,
)

from std_msgs.msg import (
    Header, 
    Empty,
)

from baxter_core_msgs.srv import (
    SolvePositionIK, 
    SolvePositionIKRequest,
)

class GetImage():
   def __init__(self):
      self.bridge = CvBridge()
      self.sub = rospy.Subscriber('/cameras/cam_link/image',Image, self.callback)
      self.current_img = None

   def callback(self,data):
      try:
         self.current_img = self.bridge.imgmsg_to_cv2(data,'bgr8')
      except CvBridgeError as e:
         print(e)

   def get(self):
      # rospy.init_node('getting_image',anonymous=True)
      return self.current_img

class MoveBaxter():
   def __init__(self, limb, verbose=True):
      self._limb_name = limb
      self._verbose = verbose
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
   
   def get_current_pose(self):
      '''returns 3 values for cartesian position, 4 values for orientation'''
      pose = self._limb.endpoint_pose()
      return pose['position'].x, pose['position'].y, pose['position'].z, pose['orientation'].x, pose['orientation'].y, pose['orientation'].z, pose['orientation'].w
   
   def get_joint_angles(self,pose):
      # pose = self.get_current_position() # obtain current pose cartesian[x,y,z] and orientation[x,y,z,w]
      hdr = Header(stamp=rospy.Time.now(), frame_id='base')
      ikreq =  SolvePositionIKRequest() # Baxter IK Solver
      ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
      try:
         joint_angles =  self._iksvc(ikreq) # request joint angles
      except (rospy.ServiceException, rospy.ROSException), e:
         rospy.logerr("Service call failed: %s" % (e,))
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
