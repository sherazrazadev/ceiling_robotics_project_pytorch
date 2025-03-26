# This Python file uses the following encoding: utf-8

import rosservice
import wsg_50_common.msg
import rospy
import std_srvs.srv
import wsg_50_common.srv


import actionlib
import ipa325_wsg50.msg as wsg_50
import ipa325_wsg50.srv

class control_wsg_50:
    def __init__(self):
        pass

        # function drive gripper to width position with speed. Return an ErrorCode
    def set_force(self,force):
        service_name = "/wsg_50_driver/set_force"
        rospy.wait_for_service(service_name)
        forceserv = rospy.ServiceProxy(service_name, wsg_50_common.srv.Conf())
        return forceserv(force)


    # function drive gripper to width position with speed. Return an ErrorCode
    def move(self, w, s,service_name="/wsg_50_driver/move"):
        velocity = float(s)
        width = float(w)
        rospy.wait_for_service(service_name)
        move_gripper = rospy.ServiceProxy(service_name, wsg_50_common.srv.Move())
        return move_gripper(width,velocity)

    # use of service /wsg_50_driver/homing to open the gripper
    def homing(self):
        service_name = "/wsg_50_driver/homing"
        rospy.wait_for_service(service_name)
        homingserv = rospy.ServiceProxy(service_name, std_srvs.srv.Empty())
        return homingserv()

    # use of service combination /wsg_50_driver/set_force (stator: grasp force = 80) & /wsg_50_driver/grasp to grasp object (stator grasp width = 13, speed = 10)
    def grasp(self,w,s,f):
        self.set_force(float(f))
        return self.move(float(w),float(s),"/wsg_50_driver/grasp")

    # use of service /wsg_50_driver/release to release object (stator: width = 4, speed = 10)
    def release(self,w, s):
        return self.move(float(w),float(s),"/wsg_50_driver/release")


class control_wsg_50_ipa_325:

    def homing(self):
        """ function homing
        function connect to the wsg_50 and home
        """
        #print("Bin in connect")
        self.Homeclient = actionlib.SimpleActionClient('WSG50Gripper_Homing', wsg_50.WSG50HomingAction)
        self.graspclient = actionlib.SimpleActionClient('WSG50Gripper_GraspPartAction', wsg_50.WSG50GraspPartAction)
        self.releaseclient = actionlib.SimpleActionClient('WSG50Gripper_PrePositionFingers', wsg_50.WSG50PrePositionFingersAction)
        #print("warte auf server")
        self.Homeclient.wait_for_server()
        self.graspclient.wait_for_server()
        self.releaseclient.wait_for_server()
        self.ack_fastStop()
        goal = wsg_50.WSG50HomingGoal(direction=True)
        #print ("sende Ziel")
        self.Homeclient.send_goal(goal)
        #print ("wait for result")
        self.Homeclient.wait_for_result()


    def grasp(self,w,s,f):
        self.set_force(f)
        goal = wsg_50.WSG50GraspPartGoal(width=w,speed=s)
        self.graspclient.send_goal(goal)
        #print ("wait for result")
        self.graspclient.wait_for_result()
        

    # use of service /wsg_50_driver/release to release object (stator: width = 4, speed = 10)
    def release(self,w, s):
        goal = wsg_50.WSG50PrePositionFingersGoal(width=w,speed=s, stopOnBlock=False)
        self.releaseclient.send_goal(goal)
        #print ("wait for result")
        self.releaseclient.wait_for_result()


        # function set force
    def set_force(self,force):
        service_name = "/SetForceLimit"
        rospy.wait_for_service(service_name)
        forceserv = rospy.ServiceProxy(service_name, ipa325_wsg50.srv.setForceLimit())
        return forceserv(force)

        # function ack Fast stop
    def ack_fastStop(self):
        service_name = "/AcknowledgeFastStop"
        rospy.wait_for_service(service_name)
        ackFastStop = rospy.ServiceProxy(service_name, ipa325_wsg50.srv.ackFastStop())
        return ackFastStop()

