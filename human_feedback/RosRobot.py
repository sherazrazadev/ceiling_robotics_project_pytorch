import sys
import roslaunch
import rospy
import time
import rosnode
from roslaunch.parent import ROSLaunchParent
import roslib
#roslib.load_manifest('bin_picking')
import rospy

#import bin_picking.msg as bp
import math
from math import pi
from tf.transformations import quaternion_from_euler, euler_from_quaternion

import std_msgs.msg as stdmsg
import sensor_msgs.msg as joint_msgs

import tf2_ros

class RosRobot():
    def __init__(self):
        self.run_id="d1a88d4a-40a2-11ec-916b-f5dc9936ea76" # string aus der Terminal ausführung herauskopiert
        self.start_roscore()
        self.set_param()


    def start_node(self):
        rospy.init_node('CEILING', anonymous=False)


    def start_launch(self, package, executable, nodename,*args):
        # http://wiki.ros.org/roslaunch/API%20Usage

        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)
        self.cli_args1 = [package, executable]
        # anhaengen der optimalen Parameter jenachdem wieviel Argumente vergeben werden sollen
        for ar in args:
              print (ar)
              self.cli_args1.append(ar)

        print (self.cli_args1)
        self.roslaunch_file1 = roslaunch.rlutil.resolve_launch_arguments(self.cli_args1)[0]
        self.roslaunch_args1 = self.cli_args1[2:]

        self.launch_files = [(self.roslaunch_file1, self.roslaunch_args1)]

        self.parent = roslaunch.parent.ROSLaunchParent(self.uuid, self.launch_files)
        self.parent.start()

        return self.parent

        # Stoppen und schließen des Knoten
    def stop_launch(self, uuid):
        print(uuid)
        uuid.shutdown()
        return None

    # function start the roscore
    def start_roscore(self):
        self.parent = ROSLaunchParent(self.run_id, [], is_core=True)     # run_id can be any string
        self.parent.start()
        time.sleep (0.5)
    # stop the roscore
    def stop_roscore(self):
        self.parent.shutdown()
        time.sleep (0.5)

    def get_ros_nodes(self):
        self.nodes = rosnode.get_node_names()
        print (self.nodes)


    def set_param(self):
        #robot_name = rospy.set_param('iiwa/toolName', 't_Vacuum')
        robot_name = rospy.set_param('iiwa/toolName', 't_WSG50')
        #robot_model = rospy.set_param('robot_model','iiwa7')
        #robot_hardware_interface = rospy.set_param('hardware_interface','PositionJointInterface')
        #robot_controllers = rospy.set_param('controllers','joint_state_controller PositionJointInterface_trajectory_controller')
        #robot_simulate = rospy.set_param('robot_simulate','false')
        #camera_stream_position = rospy.set_param('/iiwa/iiwa/camera1/camera_info/camera_stream_pos_z','0 0 5.0 1 0 0 0 world camera_stream 10')
        #robot_description = rospy.set_param('robot_description','/iiwa/robot_description_kinematics/manipulator/kinematics_solver_attempts')
        #self.robot_serial_port = rospy.set_param('~port','/dev/ttyACM0')
        #self.robot_serial_port2 = rospy.set_param('/dev/ttyUSB0','/dev/ttyACM0')
        x=12


    def init_robot_tool(self):
        self.cmd_pub_tool = rospy.Publisher('toggle_valve', stdmsg.Int8, queue_size=1)


    def control_tool(self,grip,release):
        if (grip==True):
            print("Vakumm an")
            self.cmd_pub_tool.publish(1)
        elif (release == True):
            print("Vakumm aus")
            self.cmd_pub_tool.publish(0)
        time.sleep(0.5)


    def shutdown_ros_node(self):
        #rospy.spin()
        time.sleep(0.5)
        rospy.on_shutdown(self.shutdown_reason)

    def shutdown_reason(self):
        print ("shutdown time!")
