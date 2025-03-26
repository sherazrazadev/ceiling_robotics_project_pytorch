import threading
import time
import wandb
import torch
import cv2 #
from argparse import ArgumentParser
from custom_env import CustomEnv
from models_evaluation import Policy
from human_feedback import human_feedback
from utils import TrajectoriesDataset  # noqa: F401
import csv
import os
import datetime
from utils import (
    # device,
    KeyboardObserver,
    MetricsLogger,
    loop_sleep,
    set_seeds,
)
import sys
import traceback

cap = cv2.VideoCapture(4) #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_step(policy, replay_memory, metrics_logger, config, stop_flag):
    while not stop_flag.isSet():
        batch = replay_memory.sample(config["batch_size"])
        camera_batch, proprio_batch, action_batch, feedback_batch = batch
        training_metrics = policy.update_params(
            camera_batch, proprio_batch, action_batch, feedback_batch
        )
        wandb.log(training_metrics)
        if not metrics_logger.empty():
            wandb.log(metrics_logger.pop())
    return


def run_env_simulation(env, policy, replay_memory, metrics_logger, config):
    keyboard_obs = KeyboardObserver()
    action_server_error = 0
    avg_pt_step = avg_pt_movement = time_episode_all_step = time_episode_robot_cam = spend_time_step = spend_time_movement_step = steps = 0
    print("444444444444")
    for episode in range(config["episodes"]):
        #avg_pt_step = time_episode_all_step / (steps + 1)
        #avg_pt_movement = time_episode_robot_cam / (steps + 1)
        #now= datetime.datetime.now()
        #formatDate_hour_min_sec = now.strftime("%H_%M_%S")
        #os.chdir('/home/faps/CEILing/CEILing256_v2/data/' + config["task"])
        #with open('process_information_feedback_train.csv', mode='a',newline='') as results_file:
        #        results_file_writer = csv.writer(results_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        #        results_file_writer.writerow([formatDate_hour_min_sec,episode,steps,avg_pt_step,avg_pt_movement,action_server_error])
        #avg_pt_step = avg_pt_movement = time_episode_all_step = time_episode_robot_cam = spend_time_step = spend_time_movement_step = steps = 0
        
        done = False
        reward = 0 # relevant sonst wird whileschleife uebersprungen
        lstm_state = None
        camera_obs, proprio_obs = env.reset()
        print("Starten der neuen Episode in 5 Sekunden (Episode): ", episode) # ANPASSUNG gmeiner 07.02
        print("Beenden jeder Zeit mit 'P'")
        print("actionserver error:", action_server_error)
        time.sleep(5)
        #keyboard_obs.wait_new_episode() # ANPASSUNG gmeiner 07.02
        keyboard_obs.reset()
        env.set_reward_zero()
        env.set_done_false()
        #print("reward at the beginning of the episode:", reward) 
        print("6999999999999999")

        while not done and reward != 1.0 and metrics_logger.episode_steps < 600:  # i.e. 15 seconds
            start_time = time.time()
            start_step_time = time.process_time()
            ret, frame = cap.read()#
            cv2.namedWindow("realsenseD435i", cv2.WINDOW_NORMAL) #prevent automated resizing of the window
            cv2.resizeWindow("realsenseD435i", 640, 960) #resize window to specified dimension
            cv2.moveWindow("realsenseD435i", 1000, 500) #move window
            frame = cv2.resize(frame, (256,256)) #width, height
            cv2.imshow("realsenseD435i", frame) #show window
            if cv2.waitKey(1) == ord('p'): #
                break #
            action, lstm_state = policy.predict(camera_obs, proprio_obs, lstm_state)
            #print("lstm_state", lstm_state)#
            print("action von lstm_state", action)
            #print("proprio_obs", proprio_obs)#
            steps = metrics_logger.episode_steps
            print("Episodenschritt x von 600:", steps)#
            if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update_feedbacktrain():
                # print("keyboard_obs.has_joints_cor()", keyboard_obs.has_joints_cor())
                # print("keyboard_obs.has_gripper_update_feedbacktrain()", keyboard_obs.has_gripper_update_feedbacktrain())
                # print("gripperupdate result", keyboard_obs.has_gripper_update())
                action, feedback = human_feedback(
                    keyboard_obs, action, config["feedback_type"]
                )
                # action[-1] = keyboard_obs.get_gripper() #change to adapt manual gripper value
            print("966666666666")
            print("action nach human_feedback", action)
            #print("reward VALUE vor env.step", reward)
            start_robot_cam_time = time.process_time()
            print("action vor step", action)
            next_camera_obs, next_proprio_obs, reward, done = env.step(action, cap) #
            spend_time_movement_step = time.process_time() - start_robot_cam_time
            #print("reward VALUE NACH env.step", reward)

            if not env.move_client_kuka.get_touchingground():
                replay_memory.add(camera_obs, proprio_obs, action, [feedback])
                camera_obs, proprio_obs = next_camera_obs, next_proprio_obs
                #print("reward before metrics logger:", reward)
                #print("reward VALUE in ENV episode reached button:", env.reward)
                #print("reward VALUE in vor log_step:", reward)
                metrics_logger.log_step(reward, feedback)
                #print("reward VALUE in NACH log_step:", reward)
                #loop_sleep(start_time)
                #print("moveclient_reset ",keyboard_obs.reset_move_client_button)
                #print("reset_button" ,keyboard_obs.reset_button)

            if keyboard_obs.reset_button:
                replay_memory.reset_current_traj()
                env.resetRoboPos()
                camera_obs, proprio_obs = env.reset()
                metrics_logger.reset_episode()#
                #env.set_done_true()
                #done = True
                print("Starten der neuen Episode in 5 Sekunden", episode)
                #keyboard_obs.wait_new_episode()
                time.sleep(5)
                print("wait episode sucessfull")
                keyboard_obs.reset()
                #print("reset sucessfull")
                env.set_reward_zero()
                
            else:
                loop_sleep(start_time)

            if keyboard_obs.episode_reached_button: # ANPASSUNG gmeiner 07.02
                env.set_done_true()
                #done = True # ANPASSUNG Gmeiner 07.02 wichtig fuer Reward
                env.set_reward_true()
                #print("reward VALUE in ENV episode reached button:", env.reward)
                #print("reward VALUE in feedbacktrain episode reached button:", reward)

                #replay_memory.save_current_traj() #Ende While-Schleife
                #metrics_logger.log_episode(episode) #Ende While-Schleife 
                #print("Starten der neuen Episode mit Taste 'N', aktuelle Episode: ", episode) # ANPASSUNG gmeiner 07.02
                #print("Beenden jeder Zeit mit 'P'")
                #print("actionserver error:", action_server_error)
                #keyboard_obs.wait_new_episode() # ANPASSUNG gmeiner 07.02
                #keyboard_obs.reset() 
                #env.set_reward_zero()
                #env.set_done_false()

            if keyboard_obs.reset_move_client_button:
                #env.move_client_kuka.connect()
                keyboard_obs.reset_move_client_button = False
                replay_memory.reset_current_traj()
                env.resetRoboPos()
                camera_obs, proprio_obs = env.reset()
                metrics_logger.reset_episode()
                print("Starten der neuen Episode in 5 Sekunden:", episode)
                #keyboard_obs.wait_new_episode()
                time.sleep(5)
                env.set_reward_zero()
                keyboard_obs.reset()
                action_server_error += 1
                #print("action server error added", action_server_error)

            if keyboard_obs.stop_program_client_button:
                print("stop")

                env.resetRoboPos()
                break

            spend_time_step = time.process_time() - start_step_time
            time_episode_all_step = time_episode_all_step + spend_time_step
            time_episode_robot_cam = time_episode_robot_cam + spend_time_movement_step

            print("Dauer fuer einen Schritt", spend_time_step)
            print("Dauer fuer einen Robot/Cam-Schritt", spend_time_movement_step)
            print("Dauer fuer eine Episode", time_episode_all_step)
            print("Dauer fuer eine Episode Robot/Cam-Schritt", time_episode_robot_cam)
        
        avg_pt_step = time_episode_all_step / (steps + 1)
        avg_pt_movement = time_episode_robot_cam / (steps + 1)
        now= datetime.datetime.now()
        formatDate_hour_min_sec = now.strftime("%H_%M_%S")
        os.chdir('/home/faps/CEILing/CEILing256_v2/data/' + config["task"])
        with open('process_information_feedback_train.csv', mode='a',newline='') as results_file:
                results_file_writer = csv.writer(results_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                results_file_writer.writerow([formatDate_hour_min_sec,episode,steps,avg_pt_step,avg_pt_movement,action_server_error])
        avg_pt_step = avg_pt_movement = time_episode_all_step = time_episode_robot_cam = spend_time_step = spend_time_movement_step = steps = 0        
        
        replay_memory.save_current_traj() #Ende While-Schleife
        metrics_logger.log_episode(episode) #Ende While-Schleife  

        if episode == 24 or episode == 49 or episode == 74:
            print(f"Saving ceiling policy with {episode} training episodes, please wait 1 minute" )

            time.sleep(60)

            file_name = "/home/faps/CEILing/CEILing256_v2/data/" + config["task"] + "/" + config["feedback_type"] + str(episode) + "_policy.pt"

            print("saving policy file")
            torch.save(policy.state_dict(), file_name)
            print("policy file saved")

        if keyboard_obs.stop_program_client_button:
            print("stop")
            env.move_client_kuka.stop_realtime()
            env.move_client_kuka.disconnect()
            #env.resetRoboPos()
            break  

    print("Anzahl Actionserver Fehler:", action_server_error)
    metrics_logger.log_session()
    return


def main(config):

    try: 
        replay_memory = torch.load("data/" + config["task"] + "/demos_10.dat")
        env = CustomEnv(config)
        robot_instance = env.get_move_client_instance()
        csv_header = ['time','episode','amount_steps','average_step_process_time','average_movement_process_time', 'action_server_errors',]
        if not os.path.exists('/home/faps/CEILing/CEILing256_v2/data/'+ config["task"] + '/process_information_feedback_train.csv'): # create a directory to sort the results of process
                os.chdir('/home/faps/CEILing/CEILing256_v2/data/' + config["task"])
                with open('process_information_feedback_train.csv', mode='a',newline='') as results_file:
                    results_file_writer = csv.writer(results_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                    results_file_writer.writerow(csv_header)
        pass 
        for i in range(50): #to get a real images after a view frame messages
            ret, frame = cap.read()#
            if frame is not None:#
                frame = cv2.resize(frame, (256,256)) #width, height
                cv2.imwrite('startimage_feedback_train.jpg',frame)#       
        print("Ausgangsbild fertig")#
        env.image_resize_transpose(frame)#
        policy = Policy(config).to(device)
        model_path = "data/" + config["task"] + "/" + "pretraining_policy.pt"
        policy.load_state_dict(torch.load(model_path))
        policy.train()
        wandb.watch(policy, log_freq=100)
        metrics_logger = MetricsLogger()
        stop_flag = threading.Event()
        training_loop = threading.Thread(
            target=train_step,
            args=(policy, replay_memory, metrics_logger, config, stop_flag),
        )
        training_loop.start()
        time.sleep(5)
        run_env_simulation(env, policy, replay_memory, metrics_logger, config)
        time.sleep(60)
        stop_flag.set()
        file_name = "/home/faps/CEILing/CEILing256_v2/data/" + config["task"] + "/" + config["feedback_type"] + "_policy.pt"
        torch.save(policy.state_dict(), file_name)
        cap.release()#
        cv2.destroyAllWindows()#
    
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

    finally:
        robot_instance.disconnect()


if __name__ == "__main__":
    set_seeds(1996)
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--feedback_type",
        dest="feedback_type",
        default="ceiling_full",
        help="options: evaluative, dagger, iwr, ceiling_full, ceiling_partial",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="CloseMicrowave",
        help="options: ApproachCableConnector, ApproachCableStrand, ApproachStator, GraspCableConnector, GraspStator, PickUpStatorReal",
    )
    args = parser.parse_args()
    config_defaults = {
        "feedback_type": args.feedback_type,
        "task": args.task,
        "episodes": 100,
        "static_env": False,
        "headless_env": False,
        "proprio_dim": 8,
        "action_dim": 7,
        "visual_embedding_dim": 256,
        "learning_rate": 3e-4,
        "weight_decay": 3e-6,
        "batch_size": 8,#16
    }
    wandb.init(config=config_defaults, project="ceiling", mode="disabled")
    config = wandb.config  # important, in case the sweep gives different values
    main(config)
