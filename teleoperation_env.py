import os
import time
import torch
import numpy as np
from argparse import ArgumentParser
from human_feedback import correct_action
from utils import KeyboardObserver, TrajectoriesDataset, loop_sleep
from custom_env import CustomEnv
from move_client import move_client
import threading
from RosRobot import RosRobot
from control_wsg_50 import control_wsg_50
import cv2
import datetime
import csv
import sys
import traceback
def main(config):
           
    
    save_path = "/home/faps/CEILing/CEILing256_v2/data/" + config["task"] + "/"
    assert os.path.exists(save_path)
    env = CustomEnv(config)
    robot_instance = env.get_move_client_instance()
    try: 
        keyboard_obs = KeyboardObserver()
        replay_memory = TrajectoriesDataset(config["sequence_len"])
        cap = cv2.VideoCapture(4)
        # print(cap.read())
        # print("debug*************************************")
        csv_header = ['time','episode','amount_steps','average_step_process_time','average_movement_process_time', 'action_server_errors',]
            
        if not os.path.exists('/home/faps/CEILing/CEILing256_v2/data/'+ config["task"] + '/process_information_teleoperation.csv'): # create a directory to sort the results of process
                os.chdir('/home/faps/CEILing/CEILing256_v2/data/' + config["task"])
                with open('process_information_teleoperation.csv', mode='a',newline='') as results_file:
                    results_file_writer = csv.writer(results_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                    results_file_writer.writerow(csv_header)
        pass
    
        for i in range(50): #to get a real images after a view frame messages
            ret, frame = cap.read()
            if frame is not None:
                #print("startframe wurde ubermittelt Nummer:", i)
                frame = cv2.resize(frame, (224,224)) #width, height
                cv2.imwrite('startimage_teleoperation.jpg',frame)
                #cv2.imwrite('startimage' + str(i) + '.jpg', frame)
        #if cap.isOpened(): #to get the first image -> but it turned out green for whatever reason
        #    ret, frame = cap.read()
        #    cap.release()
        #    if ret and frame is not None:
        #        print("startframe wurde ubermittelt")
        #        cv2.imwrite('startimage.jpg', frame)
        #cap.release()
        env.image_resize_transpose(frame)
        #print("uebergebenes frame:", frame)    
        camera_obs, proprio_obs = env.reset()
        keyboard_obs.set_gripper(0.9)
        gripper_open = 0.9
        gripper_last_state = 0.9
        action_server_error = 0
        #cap = cv2.VideoCapture(4)
        time.sleep(1)
        print("Start der ersten Episode mit Taste 'N'(Keyboard) oder 'START'(Gamepad)") # ANPASSUNG gmeiner 07.02
        keyboard_obs.wait_new_episode() # ANPASSUNG gmeiner 07.02
        #input( "Start der ersten Episode mit Taste 'N'(Keyboard) oder 'START'(Gamepad)")
        keyboard_obs.reset()
        episodes_count = 0
        avg_pt_step = avg_pt_movement = time_episode_all_step = time_episode_robot_cam = spend_time_step = spend_time_movement_step = steps = 0
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])#
        while episodes_count < config["episodes"]:
            start_time = time.time()
            start_step_time = time.process_time()
            ret, frame = cap.read()#
            cv2.namedWindow("realsenseD435i", cv2.WINDOW_NORMAL) #prevent automated resizing of the window
            cv2.resizeWindow("realsenseD435i", 640, 960) #resize window to specified dimension
            cv2.moveWindow("realsenseD435i", 1000, 500) #move window
            frame = cv2.resize(frame, (224,224)) #width, height
            cv2.imshow("realsenseD435i", frame) #show window
            #cap.release()
            #now= datetime.datetime.now()
            #formatDate_imagename = now.strftime("%Y_%m_%d_%H_%M_%S")
            #cv2.imwrite(os.path.join(path , 'image_'+str(formatDate_imagename)+ 'episode:' +str(episodes_count) + '.jpg'), frame)
            #env.image_resize_transpose(frame)#
            if cv2.waitKey(1) == ord('p'):
                break
            #key = cv2.waitKey(1)

            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_open])#
            #print("action nach null setzen", action[-1])        
            if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
                action = correct_action(keyboard_obs, action)
                #print(action)
                gripper_open = action[-1]
            action[-1] = keyboard_obs.get_gripper() ## immer -0.9 oder +0.9
            #print("action nach get_gripper", action[-1]) 

            start_robot_cam_time = time.process_time()
            next_camera_obs, next_proprio_obs, reward, done = env.step(action, cap)
            spend_time_movement_step = time.process_time() - start_robot_cam_time
            #print("action before replay_memory_add:", action)
            #print("proprio_obs before replay_memory_add:", next_proprio_obs)
            replay_memory.add(camera_obs, proprio_obs, action, [1])

            #print("camera_obs",camera_obs)
            #print("camera_obs_datatype", type(camera_obs))
            #print("camera_obs_dtype",camera_obs.dtype)

            #print("proprio_obs",proprio_obs)
            #print("proprio_obs_datatype", type(proprio_obs))
            #print("proprio_obs_dtype",proprio_obs.dtype)

            #print("action",action)
            #print("action_datatype", type(action))
            #print("action_dtype",action.dtype)

            camera_obs, proprio_obs = next_camera_obs, next_proprio_obs
            
            
            if keyboard_obs.reset_button:
                replay_memory.reset_current_traj()
                env.resetRoboPos()
                camera_obs, proprio_obs = env.reset()
                gripper_open = 0.9
                print("Starten der neuen Episode mit Taste 'N/Start', aktuelle Episode: ", episodes_count)
                keyboard_obs.wait_new_episode()
                env.set_reward_zero()
                keyboard_obs.reset()

            

            elif done: #nochmal anpassen i. V. zu feedbacktrain 26.04 MG
                replay_memory.save_current_traj()
                camera_obs, proprio_obs = env.reset()
                gripper_open = 0.9
                episodes_count += 1
                print("Starten der neuen Episode mit Taste 'N', aktuelle Episode: ", episodes_count) # ANPASSUNG gmeiner 07.02
                print("Beenden jeder Zeit mit 'P'")
                keyboard_obs.wait_new_episode() # ANPASSUNG gmeiner 07.02
                keyboard_obs.reset()
                avg_pt_step = time_episode_all_step / (steps + 1)
                avg_pt_movement = time_episode_robot_cam / (steps + 1)
                now= datetime.datetime.now()
                formatDate_hour_min_sec = now.strftime("%H_%M_%S")
                os.chdir('/home/faps/CEILing/CEILing256_v2/data/' + config["task"])
                with open('process_information_teleoperation.csv', mode='a',newline='') as results_file:
                        results_file_writer = csv.writer(results_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                        results_file_writer.writerow([formatDate_hour_min_sec,episodes_count,steps,avg_pt_step,avg_pt_movement,action_server_error])
                avg_pt_step = avg_pt_movement = time_episode_all_step = time_episode_robot_cam = spend_time_step = spend_time_movement_step = steps = 0
                env.set_reward_zero()
                done = False
                env.set_done_false()
                steps = 0
            else:
                loop_sleep(start_time)
                
            if keyboard_obs.episode_reached_button: # ANPASSUNG gmeiner 07.02
                        env.set_done_true()
                        done = True # ANPASSUNG Gmeiner 07.02
                        env.set_reward_true()

            if keyboard_obs.reset_move_client_button:
                # env.move_client_kuka.disconnect()
                # env.move_client_kuka.connect()
                keyboard_obs.reset_move_client_button = False
                replay_memory.reset_current_traj()
                env.resetRoboPos()
                camera_obs, proprio_obs = env.reset()
                gripper_open = 0.9
                print("Starten der neuen Episode mit Taste 'N/Start', aktuelle Episode: ", episodes_count)
                keyboard_obs.wait_new_episode()
                env.set_reward_zero()
                keyboard_obs.reset()
                action_server_error += 1
                steps= 0

            if keyboard_obs.stop_program_client_button:
                 print("stop")
                 env.resetRoboPos()
                 break
            
            # if episodes_count % 20 == 0 and episodes_count >= 20 :
            #     file_name = "intermediate_" + str(episodes_count) + ".dat"
            #     torch.save(replay_memory, save_path + file_name)
                 

            spend_time_step = time.process_time() - start_step_time
            time_episode_all_step = time_episode_all_step + spend_time_step
            time_episode_robot_cam = time_episode_robot_cam + spend_time_movement_step
            steps += 1
            #print("Dauer fuer einen Schritt", spend_time_step)
            #print("Dauer fuer einen Robot/Cam-Schritt", spend_time_movement_step)
            #print("Dauer fuer eine Episode", time_episode_all_step)
            #print("Dauer fuer eine Episode Robot/Cam-Schritt", time_episode_robot_cam)

            
                
                
        # a = input("Do you want save the demos ? press ö")
        # if a == 'ö':
        print("Anzahl Actionserver Fehler:", action_server_error)     
        file_name = "demos_" + str(config["episodes"]) + ".dat"
        if config["save_demos"]:
            print('saving demos')
            torch.save(replay_memory, save_path + file_name)
            print('saved demos')
            cap.release()
            cv2.destroyAllWindows()
            
        return
    
    except BaseException as ex:
        # Get current system exception
        ex_type, ex_value, ex_traceback = sys.exc_info()

        # Extract unformatter stack traces as tuples
        trace_back = traceback.extract_tb(ex_traceback)

        # Format stacktrace
        stack_trace = list()

        for trace in trace_back:
            stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

        print("Exception type : %s " % ex_type.__name__)
        print("Exception message : %s" %ex_value)
        print("Stack trace : %s" %stack_trace)


        print("Error")
        print("saving intermediate file:", episodes_count)     
        file_name = "intermediate_end" + str(episodes_count) + ".dat"
        if config["save_demos"]:
            torch.save(replay_memory, save_path + file_name)
            cap.release()
            cv2.destroyAllWindows()

    finally:
        robot_instance.disconnect()
         



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="CloseMicrowave",
        help="options: ApproachCableConnector, ApproachCableStrand, ApproachStator, GraspCableConnector, GraspStator, PickUpStatorReal",
    ) # add tasks (keep in mind to create folders at directory data)
    args = parser.parse_args()
    config = {
        "task": args.task,
        "static_env": False,
        "headless_env": False,
        "save_demos": True,
        "episodes": 50,
        "sequence_len": 150,
    }
    main(config)
