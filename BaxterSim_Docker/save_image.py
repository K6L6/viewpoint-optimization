#!/usr/bin/env python

import ipdb
import os
import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

class SaveImage():
    def __init__(self,directory):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/cameras/cam_link/image",Image,self.callback)
        self.directory = directory
        # self.object_number = object_number
        self.file_number = -1
        self.last_saved_file_number = -1
        self.snap_flag = False

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
            if self.snap_flag == True:
                cv2.imwrite(self.directory+"image_{}.png".format(self.file_number),cv_image)
                self.last_saved_file_number = self.file_number
                self.snap_flag = False

        except CvBridgeError as e:
            print(e)

def main(args):
    directory = "/home/baxter_ws/images/"
    im = SaveImage(directory,0)
    rospy.init_node("capture_images",anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Exited...")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main(sys.argv))