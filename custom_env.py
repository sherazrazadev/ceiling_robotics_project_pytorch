#from time import time
import torch #
from matplotlib import image
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from utils import euler_to_quaternion #KeyboardObserver #,task_switch #rausgenommen funktioniert, aber erstmal auskommentiert auch bei utils
from move_client import move_client
from move_client import move_client_iiwa_python
import threading
from RosRobot import RosRobot
from control_wsg_50 import control_wsg_50 ,control_wsg_50_ipa_325
import rospy
import iiwa_msgs.msg as iiwa_msgs
from math import pi, sqrt
import time
import cv2
import os
import datetime
import math

class GripperPlot:
    def __init__(self, headless):
        self.headless = headless
        if headless:
            return
        self.displayed_gripper = 0.9
        print("before plot figure")
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        print("after plot figure")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        horizontal_patch = plt.Rectangle((-1, 0), 2, 0.6)
        self.left_patch = plt.Rectangle((-0.9, -1), 0.4, 1, color="black")
        self.right_patch = plt.Rectangle((0.5, -1), 0.4, 1, color="black")
        ax.add_patch(horizontal_patch)
        ax.add_patch(self.left_patch)
        ax.add_patch(self.right_patch)
        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(0.1)
        for _ in range(2):
            self.set_data(0)
            plt.pause(0.1)
            self.set_data(1)
            plt.pause(0.1)
        return

    def set_data(self, last_gripper_open):
        #print("lastgripperopen:", last_gripper_open)
        if self.headless:
            return
        if self.displayed_gripper == last_gripper_open:
            return
        if last_gripper_open == 0.9:
            #print("lastgripperopen", last_gripper_open)
            self.displayed_gripper = 0.9
            self.left_patch.set_xy((-0.9, -1))
            self.right_patch.set_xy((0.5, -1))
        elif last_gripper_open == -0.9:
            self.displayed_gripper = -0.9
            self.left_patch.set_xy((-0.4, -1))
            self.right_patch.set_xy((0, -1))
        self.fig.canvas.draw()
        plt.pause(0.01)
        return

    def reset(self):
        self.set_data(0.9) #von 1 auf 0.9 gesetzt um Greifergrafik zu schließen Gmeiner 17.3


class CustomEnv:
    def __init__(self, config):
        # image_size=(128, 128) & Roboterinitialisierung
        self.robot = RosRobot() # ANPASSUNG gmeiner
        self.robot.start_node() # ANPASSUNG gmeiner 
        print("start ros node")	
        #self.control_wsg50 = control_wsg_50() # ANPASSUNG gmeiner
        self.robot.start_launch('ipa325_wsg50',"wsg50.launch",'wsg_50')
        self.control_wsg50 = control_wsg_50_ipa_325()
        self.control_wsg50.homing()
        print("start wsg 50 gripper")
        time.sleep(5)
        #self.move_client_kuka = move_client() # ANPASSUNG gmeiner

        self.move_client_kuka = move_client_iiwa_python() # commented for testing purpose
        #self.t_pos_start = threading.Thread(target=self.move_client_kuka.get_currentposition) # ANPASSUNG gmeiner
        #self.s_pos_start = threading.Thread(target=self.subscriber)
        #self.s_pos_start.start()
        #self.t_pos_start.start() # ANPASSUNG gmeiner
        self.move_client_kuka.connect() # ANPASSUNG gmeiner # commented for testing purpose
        #self.move_client_kuka.move(-0.185,-0.65,0.355,180,0,90,"PTP",0,2,93) # self,x,y,z,roll ,pitch ,yaw ,movetyp, e1 , status ,turn 
        self.resetRoboPos() # commented for testing purpose

        self.move_client_kuka.get_currentposition() # commented for testing purpose
        print("Start Realtime")
        self.move_client_kuka.start_realtime() # commented for testing purpose
        
        #self.robot.start_launch('wsg_50_driver',"wsg_50_tcp_script.launch",'wsg_50')
        #self.gripper_plot = GripperPlot(config["headless_env"])
        print("init gripper plot")
        self.gripper_open = 0.9
        self.gripper_last_state = 0.9
        self.max_delay_gripper_steps = 20 # vorher 20 statt 5 # changed to 20 from 5
        self.gripper_deque = deque([0.9] * self.max_delay_gripper_steps, maxlen=self.max_delay_gripper_steps) # vorher 20 statt 5 # changed to 20 from 5
        self.frame = 0
        self.frame_new = 0
        self.reward = 0
        self.done = False
        self.path = '/home/faps/CEILing256_v2/Test_Images'
        print("init gripper plot finish")
        return
    
    def get_move_client_instance(self):
        return self.move_client_kuka
 
    #def subscriber(self):        not needed as using iiwapy joint values are accessed rather than ros
    #     rospy.Subscriber("/iiwa/state/JointPosition", iiwa_msgs.JointPosition, self.jointPosition_callback)
        
    #     return
    
    # def jointPosition_callback(self,msg):
    #     """ jointPosition_callback
    #     callback Function to save the current Joint Pos in class variables
    #     """
    #     self.Joint1 = self.rad2deg(msg.position.a1)
    #     self.Joint2 = self.rad2deg(msg.position.a2)
    #     self.Joint3 = self.rad2deg(msg.position.a3)
    #     self.Joint4 = self.rad2deg(msg.position.a4)
    #     self.Joint5 = self.rad2deg(msg.position.a5)
    #     self.Joint6 = self.rad2deg(msg.position.a6)
    #     self.Joint7 = self.rad2deg(msg.position.a7)        
    #     self.arJoint = [self.Joint1, self.Joint2, self.Joint3, self.Joint4, self.Joint5, self.Joint6, self.Joint7]
    #     self.np_arJoint = np.array(self.arJoint)
        #verified joints (-105, 41, 0, -43, 0, 95, -15 in start position)

    def rad2deg(self,a):
        a=float(a)
        return ((a*360)/(2*pi))
        
    def image_resize_transpose(self, frame):
        self.frame = cv2.resize(frame, (224, 224))
        self.frame_new = np.transpose(self.frame, (2,0,1))
        return

    #def image_resize_transpose(self, frame):
    #	self.frame = cv2.resize(frame, (128,128))
    #    self.frame_new = np.transpose(self.frame, (2, 0, 1))
    #    return 
    
    def reset(self):
        #self.gripper_plot.reset()
        self.gripper_open = 0.9
        #action_delayed = self.postprocess_action(action)

        if self.gripper_last_state != 1: #0.9?
            #self.control_wsg50.release(109,100) # ANPASSUNG Gmeiner 07.02 for stator
            # self.control_wsg50.release(50,50) # for socket, wire
            self.control_wsg50.release(1.0,50) # for stator inside picking
            pass
        self.gripper_deque = deque([0.9] * self.max_delay_gripper_steps, maxlen=self.max_delay_gripper_steps) #20 old wie viele 0.9 oder -0.9 bis zum Ausführen
        self.resetRoboPos()
        print("Start Realtime")
        try:
            self.move_client_kuka.start_realtime()
        except:
            self.move_client_kuka.disconnect()
            input('Restart MatlabToolboxServer application in smartpad and press enter key \n') 
            self.move_client_kuka.connect()
            self.move_client_kuka.get_currentposition()
            self.move_client_kuka.start_realtime()
        camera_obs, proprio_obs = self.obs_split()
        return camera_obs, proprio_obs #reset fuer kamera etc abaendern

    def resetRoboPos(self):
        print("Stop Realtime")
        try:
            self.move_client_kuka.stop_realtime()
        except:
            self.move_client_kuka.disconnect()
            input('Restart MatlabToolboxServer application in smartpad and press enter key \n') 
            self.move_client_kuka.connect()
        #self.move_client_kuka.move([0, 0, 0, -math.pi / 2, 0, math.pi / 2, 0],[0.1])
        # closer positioning of eef to the object to be grasped
        print("Moving to initial position")

        try:
            # self.move_client_kuka.move([0, 0.285, 0, -1.5673, 0, 1.2666, 0],[0.1]) # for stator picking position
            self.move_client_kuka.move([-0.17, 0.2857, 0, -1.175, 0.004, 1.657, -0.1695],[0.1]) # multiple stators position
            # self.move_client_kuka.move([0, 0.4475, 0, -1.76418, 0, 0.9070, 0],[0.1])  # for socket ,wire picking position
        except:
            self.move_client_kuka.disconnect()
            input('Restart MatlabToolboxServer application in smartpad and press enter key \n') 
            self.move_client_kuka.connect()
            #self.move_client_kuka.move([0, 0.285, 0, -1.5673, 0, 1.2666, 0],[0.1]) # for stator picking position
            self.move_client_kuka.move([-0.17, 0.2857, 0, -1.175, 0.004, 1.657, -0.1695],[0.1]) # multiple stators position
            # self.move_client_kuka.move([0, 0.4475, 0, -1.76418, 0, 0.9070, 0],[0.1])  # for socket, wire picking position

        print("requesting current position")

        # self.move_client_kuka.current_eff_pos = [510.4604296963686, -0.01905945632298702, 294.52175421125526, -3.1415468518094034, 0.022709505661093442, -3.1415471522999496]
        # print(f"current position {self.move_client_kuka.current_eff_pos}")
        try:
            cp = self.move_client_kuka.get_currentposition()
            print(f"current position {cp}")
        except:
            self.move_client_kuka.disconnect()
            input('Restart MatlabToolboxServer application in smartpad and press enter key \n') 
            self.move_client_kuka.connect()
            cp = self.move_client_kuka.get_currentposition()
            print(f"current position {cp}")

            
        #self.move_client_kuka.move(0.635,-0.162,0.280,90,0,-180,"PTP",0,2,25) #latest 06.04.22 (zuvor 280 mm)
        #self.move_client_kuka.move(0.538,-0.444,0.683,139,38,178,"PTP",0,2,73) #test 06.04.22, ob simulatnes Steuern der Achsen am Achsbereich liegt-> Nein, gleiches Problem
        return
    
    def step(self, action, cap):
        action_delayed = self.postprocess_action(action)
        #print("action delayed in step:", action_delayed)
        try:
            
            if np.count_nonzero(action_delayed[0:6]) != 0:
                #print("action delayed bevor move relativ ********", action_delayed)
                
                self.move_client_kuka.move_relativ(action_delayed)
            #print("gripper_open before WSG50 action", self.gripper_open) 
            #print("gripper_last_state before WSG50 action", self.gripper_last_state) 
            if self.gripper_open == 0.0: # ANPASSUNG gmeiner 07.02
                #print("gripper griff")
                if self.gripper_open != self.gripper_last_state:
                    #self.control_wsg50.grasp(75,100,20) # ANPASSUNG Gmeiner 07.02   for stator
                    self.control_wsg50.grasp(14,50,35) # for inside stator picking
                    #self.control_wsg50.grasp(15,50,20) # for scoket picking
                    # self.control_wsg50.grasp(2,50,20) # for wire picking
                    #print("gripper griff NACH DEM GREIFBEFEHL")         
                    time.sleep(0.1)
            if self.gripper_open == 1.0: # ANPASSUNG gmeiner 07.02
                #print("bin im Gripper open IF")
                if self.gripper_open != self.gripper_last_state:
                    #self.control_wsg50.release(109,100) # ANPASSUNG Gmeiner 07.02  for stator
                    self.control_wsg50.release(1.0,50) # for stator inside picking
                    # self.control_wsg50.release(50,50) # for socket, wire
                    #print ("öffne")
                    #print("gripper oeffnete")
                    time.sleep(0.1)
            #print("gripper_open after WSG50 action", self.gripper_open)
            
            #if np.mean(action_delayed[0:6]) != 0:
            #    self.move_client_kuka.move_relativ(action_delayed)

            self.gripper_last_state = self.gripper_open
            #print("gripper_last_state before WSG50 action", self.gripper_last_state) 
        except:
            zero_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, action_delayed[-1]]
            #next_obs, reward, done = self.task.step(zero_action)
            self.move_client_kuka.move_relativ(zero_action)

        #cap = cv2.VideoCapture(4)
        ret, frame = cap.read()
        #cap.release()
        #now= datetime.datetime.now()
        #formatDate_imagename = now.strftime("%Y_%m_%d_%H_%M_%S")
        #cv2.imwrite(os.path.join(self.path , 'image_'+str(formatDate_imagename) + '.jpg'), frame)
        self.image_resize_transpose(frame)#
        camera_obs, proprio_obs = self.obs_split()
        print("reward in custom env before return:", self.reward)
        return camera_obs, proprio_obs, self.reward, self.done#reward einfuegen

    def render(self):
        return

    def set_done_true(self):
        self.done = True
        return

    def set_done_false(self):
        self.done = False
        return

    def set_reward_true(self):
        self.reward = 1.0
        return

    def set_reward_zero(self):
        self.reward = 0
        return

    def close(self):
        self.env.shutdown()
        return
    
    ## Function deg2rad
    ## conversion deg to rad
    def deg2rad(self,a):
        a=float(a)
        return (a/360)*2*pi


    def postprocess_action(self, action):
        step_rad= self.deg2rad(0.25)/0.9 #1.25 Grad in Rad -> 0.9 Schrittweite
        delta_position = action[:3] * 1 # in mm [m] zuvor war es 0.01 -> 0.09m, gewaehlt x5 (0.005) -> 4.5 mm also ************0.0045m*********** (verrechnet in moveclient)
        delta_angle_rad = action[3:6] * step_rad #zuvor 0.04 Paper->0.25°, gewahelt 0,2 und x5 ->***********1.25°******************
        #print(action[3:6] * 0.2)
        #action[3:6] = 0,9 x 0,04 -> 0,0036 zuvor
        #print("gripper action-1 vor gripper_delay", action[-1])
        gripper_delayed = self.delay_gripper(action[-1])
        #print("delayed_gripper", gripper_delayed)
        #gripper_delayed = action[-1]
        action_post = np.concatenate(
            (delta_position, delta_angle_rad, [gripper_delayed])
        )
        return action_post 

    def delay_gripper(self, gripper_action):
        if gripper_action >= 0.0:
            gripper_action = 0.9
        elif gripper_action < 0.0:
            gripper_action = -0.9
        #self.gripper_plot.set_data(gripper_action)
        self.gripper_deque.append(gripper_action)
        if all([x == 0.9 for x in self.gripper_deque]):
            self.gripper_open = 1.0
        elif all([x == -0.9 for x in self.gripper_deque]):
            self.gripper_open = 0
        #print("delayed_gripper",self.gripper_open)
        return self.gripper_open
        
    def obs_split(self):
	#get rgb image in torch order & robot and gripper joints
        #self.frame_new = np.transpose(frame, (2, 0, 1)) # Transpose it into torch order
        
        camera_obs = self.frame_new #scheint ein int object zu sein  (print("obs_split_cameraobs:", camera_obs.dtype)AttributeError: 'int' object has no attribute 'dtype')
        #camera_obs = np.transpose(frame, (2, 0, 1)) # Transpose it into torch order (CHW)
        #camera_obs = np.array(self.frame_new, dtype=np.uint8)
        #camera_obs = np.vstack(self.camera_obs).astype(np.float32)
        #b = torch.from_numpy(camera_obs)
        #print(b)
        #print("NewBegin:", camera_obs.dtype)
        #print("NewBegin:", camera_obs)
        #print("obs_split_cameraobs:", camera_obs.dtype)
        #vector = np.vectorize(np.float)
        #camera_obs = vector(camera_obs)
        #camera_obs = np.vstack(camera_obs).astype(np.float32)
        #print(camera_obs.dtype)   
        #print ("arJoint Type",type(self.np_arJoint)) 
        self.np_arJoint = self.move_client_kuka.get_current_joint_pos_deg()
        proprio_obs = np.append(self.np_arJoint, self.gripper_open) # STARTET MIT 0.9 und nicht mit 0 oder 1 ->nochmal nachschauen
        #print ("arJoint dType",self.np_arJoint.dtype)
        #print("proprio in obs_split:",proprio_obs)
        #print("gripper state obs_split:", self.gripper_open)
        #verified joints (-105, 41, 0, -43, 0, 95, -15, 0.9(Gripper) in start position) and verified camera obs
                    
        return camera_obs, proprio_obs
