#!/usr/bin/env python3

import roslib
from rospy.core import rospydebug
import time
import rospy
import actionlib
import tf
import numpy as np
import os
import csv
import datetime

#import iiwa_msgs.msg as iiwa_msgs

from math import pi
from tf.transformations import quaternion_from_euler, euler_from_quaternion


import math
import time
from datetime import datetime
# import sys
# sys.path.append('/home/faps/CEILing256_v2/src/iiwa_py3')
from iiwa_py3.iiwaPy3 import iiwaPy3


class move_client:
    def __init__(self):
        self.xPos = 0
        self.yPos = 0
        self.zPos = 0
        self.xOrient = 0
        self.yOrient = 0
        self.zOrient = 0
        self.wOrient = 0
        self.AOrient = 0
        self.BOrient = 0
        self.COrient = 0

        #self.csv_header = ['time','xPos','yPos','zPos','qx','qy','qz','qw','A_deg','B_deg','C_deg','action','actual position x','actual position y', 'actual position z', 'actual A in deg', 'actual B in deg' ,'actual C in deg x',]

        
        #if not os.path.exists('/home/faps/CEILing256_v2/move_client.csv'): # create a directory to sort the result per day
        #    os.chdir('/home/faps/CEILing')
        #    with open('move_client.csv', mode='a',newline='') as results_file:
        #        results_file_writer = csv.writer(results_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        #        results_file_writer.writerow(self.csv_header)
        #pass

    ## Function deg2rad
    ## conversion deg to rad
    def deg2rad(self,a):
        a=float(a)
        return (a/360)*2*pi


    ## Function rad2deg
    ## conversion rad to deg
    def rad2deg(self,a):
        a=float(a)
        return ((a*360)/(2*pi))


    def quat2euler(self,quat):
        euler = euler_from_quaternion(quat,axes='sxyz')
        return euler

    def euler2quat(self,orientation):
        """ function euler2quat
        function calculate euler in quaternion
        """
        return quaternion_from_euler(orientation[2],orientation[1],orientation[0],'sxyz')

    def connect(self):
        rospy.loginfo("Starting Connection to iiwa Action Server. Please start the Robot at the Smart Pad")
        self.client_lin = actionlib.SimpleActionClient('/iiwa/action/move_to_cartesian_pose_lin', iiwa_msgs.MoveToCartesianPoseAction)
        self.client_lin.wait_for_server()
        rospy.loginfo("Connection to iiwa Action Server Lin succesfull")
        self.client_ptp = actionlib.SimpleActionClient('/iiwa/action/move_to_cartesian_pose', iiwa_msgs.MoveToCartesianPoseAction)
        self.client_ptp.wait_for_server()
        rospy.loginfo("Connection to iiwa Action Server PTP succesfull")
        #print("get_currentposition:", move_client.get_currentposition(self))

    def get_currentposition(self):
        rospy.Subscriber('/iiwa/state/CartesianPose', iiwa_msgs.CartesianPose, self.pose_callback)

    def pose_callback(self,msg):
        #print(msg)
        #time.sleep(1)
        self.xPos = msg.poseStamped.pose.position.x
        self.yPos = msg.poseStamped.pose.position.y
        self.zPos = msg.poseStamped.pose.position.z

        self.xOrient = msg.poseStamped.pose.orientation.x
        self.yOrient = msg.poseStamped.pose.orientation.y
        self.zOrient = msg.poseStamped.pose.orientation.z
        self.wOrient = msg.poseStamped.pose.orientation.w
        
        self.e1 = msg.redundancy.e1
        self.status = msg.redundancy.status
        self.turn = msg.redundancy.turn

        quat = [self.xOrient,self.yOrient,self.zOrient,self.wOrient]
        euler = self.quat2euler(quat)

        self.AOrient = self.rad2deg(euler[2])
        self.BOrient = self.rad2deg(euler[1])
        self.COrient = self.rad2deg(euler[0])



    def move(self,x,y,z,A ,B ,C ,movetyp,e1 = None , status = None ,turn = None):


        #r_rad = self.deg2rad(float(roll))
        #p_rad = self.deg2rad(float(pitch))
        #y_rad = self.deg2rad(float(yaw)) #A und C vertauscht, alt mit oben roll, pitch, yaw

        #q = quaternion_from_euler(r_rad,p_rad,y_rad,'sxyz')


        q = self.euler2quat ([self.deg2rad(float(A)),self.deg2rad(float(B)),self.deg2rad(float(C))])

        goal_pose = iiwa_msgs.CartesianPose()
        goal_pose.poseStamped.pose.position.x = x
        goal_pose.poseStamped.pose.position.y = y
        goal_pose.poseStamped.pose.position.z = z
        goal_pose.poseStamped.pose.orientation.x = q[0]#0.7071349620819092
        goal_pose.poseStamped.pose.orientation.y = q[1]#0.7070785731455016
        goal_pose.poseStamped.pose.orientation.z = q[2]#9.196629941950118e-06
        goal_pose.poseStamped.pose.orientation.w = q[3]#3.0092893482687172e-05
        #goal_pose.redundancy.e1 = 0.798130263823509
        #goal_pose.redundancy.status = 2
        #goal_pose.redundancy.turn = 89
        goal_pose.poseStamped.header.stamp = rospy.Time.now()
        goal_pose.poseStamped.header.frame_id = "iiwa_link_0"


        #print(goal_pose)
        
        goal = iiwa_msgs.MoveToCartesianPoseGoal(cartesian_pose=goal_pose)
        if (movetyp == "LIN"):
            self.client_lin.send_goal(goal)
            self.client_lin.wait_for_result()
          

        elif (movetyp == "PTP"):
            # For PTP Movement use the current configuration or use the configuration of the Teachpoint
            if (e1 == None):
                goal_pose.redundancy.e1 = self.e1
            else:
                goal_pose.redundancy.e1 = self.deg2rad(e1)

            if (e1 == None):
                goal_pose.redundancy.status = self.status
            else:
                goal_pose.redundancy.status = status

            if (e1 == None):
                goal_pose.redundancy.turn = self.turn
            else:
                goal_pose.redundancy.turn = turn

            self.client_ptp.send_goal(goal)
            self.client_ptp.wait_for_result()
        


    def move_relativ(self,action):
        #print (action[3:7])

        delta_rad_angle = self.quat2euler(action[3:7])

        delta_deg_angle_B = round(self.rad2deg(delta_rad_angle[0]),2) #B
        delta_deg_angle_C = round(self.rad2deg(delta_rad_angle[1]),2) #C
        delta_deg_angle_A = round(self.rad2deg(delta_rad_angle[2]),2) #A

        #print (delta_deg_angle_B,delta_deg_angle_C,delta_deg_angle_A)

        C_deg = float(delta_deg_angle_C) + self.COrient #tausch wegen Beugen und Schwenken damals 3
        B_deg  = float(delta_deg_angle_B) * -1 + self.BOrient #tausch wegen Beugen und Schwenken damals 4, hier das Vorzeichen noch vertauscht
        A_deg = float(delta_deg_angle_A) + self.AOrient

        #roll = float(action[4]*0.277*50) + self.roll #tausch wegen Beugen und Schwenken damals 3
        #pitch  = float(action[3]*0.277*50) * -1 + self.pitch #tausch wegen Beugen und Schwenken damals 4, hier das Vorzeichen noch vertauscht
        #yaw = float(action[5]*0.277*50) + self.yaw


        quat = self.euler2quat ([self.deg2rad(float(A_deg)),self.deg2rad(float(B_deg)),self.deg2rad(float(C_deg))])
        #print (quat)
        #translation pose plus 0.9 (action) -> 0.009 mm + pose [bei 0.001]
        goal_pose = iiwa_msgs.CartesianPose()
        goal_pose.poseStamped.pose.position.x = float(action[0]) + self.xPos
        goal_pose.poseStamped.pose.position.y = float(action[1]) + self.yPos
        goal_pose.poseStamped.pose.position.z = float(action[2]) + self.zPos
        #print("goal_pose:", goal_pose)
        #orientation pose plus 0.9 (action) -> 9 Grad
        goal_pose.poseStamped.pose.orientation.x = quat[0]
        goal_pose.poseStamped.pose.orientation.y = quat[1]
        goal_pose.poseStamped.pose.orientation.z = quat[2]
        goal_pose.poseStamped.pose.orientation.w = quat[3]

        goal_pose.poseStamped.header.stamp = rospy.Time.now()
        goal_pose.poseStamped.header.frame_id = "iiwa_link_0"

        goal = iiwa_msgs.MoveToCartesianPoseGoal(cartesian_pose=goal_pose)
        rospy.loginfo("vor send goal")
        self.client_lin.send_goal(goal)
        rospy.loginfo("send goal")
        self.client_lin.wait_for_result(rospy.Duration(1))
        rospy.loginfo("get result")

        now= datetime.datetime.now()
        formatDate_hour_min_sec = now.strftime("%H_%M_%S")

        #with open('move_client.csv', mode='a',newline='') as results_file:
        #        results_file_writer = csv.writer(results_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        #        results_file_writer.writerow([formatDate_hour_min_sec,goal_pose.poseStamped.pose.position.x,goal_pose.poseStamped.pose.position.y,goal_pose.poseStamped.pose.position.z,goal_pose.poseStamped.pose.orientation.x,goal_pose.poseStamped.pose.orientation.y,goal_pose.poseStamped.pose.orientation.z,goal_pose.poseStamped.pose.orientation.w,A_deg,B_deg,C_deg,action,self.xPos,self.yPos,self.zPos,self.AOrient,self.BOrient,self.COrient])


class move_client_iiwa_python():

    def __init__(self):
        self.xPos = 0
        self.yPos = 0
        self.zPos = 0
        self.xOrient = 0
        self.yOrient = 0
        self.zOrient = 0
        self.wOrient = 0
        self.AOrient = 0
        self.BOrient = 0
        self.COrient = 0
        self.realtime_connection_stopped = True
        self.touching_ground = False
        self.current_joint_positions = np.array([0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0])

    ## Function deg2rad
    ## conversion deg to rad
    def deg2rad(self,a):
        a=float(a)
        return (a/360)*2*pi


    ## Function rad2deg
    ## conversion rad to deg
    def rad2deg(self,a):
        a=float(a)
        return ((a*360)/(2*pi))
    

     ## Function rad2deg
    ## conversion rad to deg
    def rad2deg_np(self,a):
        a=np.array(a)
        return ((a*360)/(2*pi))


    def quat2euler(self,quat):
        euler = euler_from_quaternion(quat,axes='sxyz')
        return euler

    def euler2quat(self,orientation):
        """ function euler2quat
        function calculate euler in quaternion
        """
        return quaternion_from_euler(orientation[2],orientation[1],orientation[0],'sxyz')
    
    def connect(self):
        ip = '172.31.1.147'
        # ip='localhost'
        TPCtransform = (0, 0, 205.5, 0, 0, 0)  # (x,y,z,alfa,beta,gama)
        self.iiwa = iiwaPy3(ip, TPCtransform)
        # iiwa = iiwaPy3(ip)
        
        self.iiwa.setBlueOn()
        time.sleep(2)
        self.iiwa.setBlueOff()
        
        print('connection established with robot')

        return self.iiwa
    
    def get_touchingground(self):
        return self.touching_ground
    
    def get_currentposition(self):
        self.current_eff_pos = self.iiwa.getEEFPos()
        return self.current_eff_pos

    def disconnect(self):
        try:
            self.iiwa.close()
            print("Disconnected successfully")
        except:
            print("Error could not disconnect")

    def getSecs(self):
        dt = datetime.now() - self.start_time
        secs = (dt.days * 24 * 60 * 60 + dt.seconds) + dt.microseconds / 1000000.0
        return secs

    def move(self,jpos,vel=[0.1]):
        
        try:

            self.iiwa.movePTPJointSpace(jpos,vel)
        
        except:
            self.iiwa.close()


    def start_realtime(self):
        self.iiwa.realTime_startDirectServoCartesian()
        self.realtime_connection_stopped = False
        time.sleep(0.04)

    def stop_realtime(self):
        if self.realtime_connection_stopped == True:
            message = "Already off-line"
            print(message)
            return
        self.iiwa.realTime_stopDirectServoCartesian()
        self.realtime_connection_stopped = True
        self.touching_ground = False
        time.sleep(0.04)

    def get_current_joint_pos_deg(self):
        return self.rad2deg_np(self.current_joint_positions)


    def move_relativ(self,action):

        try: 
            print("action")
            print(action)

            if self.current_eff_pos[2] <= 7.50:
                action[2] = 0
                self.touching_ground = True
            else:
                self.touching_ground = False

            

            self.current_eff_pos[0:6] = [sum(x) for x in zip(self.current_eff_pos[0:6], action[0:6])]

            # to avoid hitting the table with gripper fingers

            
            # if self.current_eff_pos[2] <= 7.00:
            #     self.current_eff_pos[2] = 7.00
            #     self.touching_ground = True 
            # else:
            #     self.touching_ground = False
             
            print('robot position to move',self.current_eff_pos)
            nn = self.iiwa.sendEEfPositionGetActualJpos(self.current_eff_pos)
            # print("First time")
            # print(nn)
            time.sleep(0.04)
            self.current_joint_positions = self.iiwa.sendEEfPositionGetActualJpos(self.current_eff_pos)
            # print("sec time")
            print('robot position after taking the action',self.current_eff_pos)

            #deg_val = np.array(self.current_joint_positions)

            print(self.rad2deg_np(self.current_joint_positions))

            # nn = self.iiwa.sendEEfPositionGetActualEEFpos(self.current_eff_pos)
            # print("First time")
            # print(nn)
            # time.sleep(0.04)
            # nn = self.iiwa.sendEEfPositionGetActualEEFpos(self.current_eff_pos)
            # print("sec time")
            # print(nn)
            
            return self.current_eff_pos    
        except:
            self.iiwa.close()

    

        
        