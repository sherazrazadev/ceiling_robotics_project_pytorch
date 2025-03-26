from audioop import avg
import torch
import wandb
import time
from models_training_attention_evaluation import Policy
from custom_env import CustomEnv
from utils import loop_sleep, set_seeds
from utils import KeyboardObserver
from argparse import ArgumentParser
import cv2
import csv
import os
import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cap = cv2.VideoCapture(4) #


def run_simulation(env, policy, episodes):
    successes = 0
    steps = 0
    action_server_error = 0
    avg_pt_step = avg_pt_movement = time_episode_all_step = time_episode_robot_cam = spend_time_step = spend_time_movement_step = 0
    keyboard_obs = KeyboardObserver()
    time.sleep(5)

    for episode in range(episodes):
        #avg_pt_step = time_episode_all_step / (steps + 1)
        #avg_pt_movement = time_episode_robot_cam / (steps + 1)
        #now= datetime.datetime.now()
        #formatDate_hour_min_sec = now.strftime("%H_%M_%S")
        #os.chdir('/home/faps/CEILing/CEILing256_v2/data/' + config["task"])
        #with open('process_information_evaluate.csv', mode='a',newline='') as results_file:
        #        results_file_writer = csv.writer(results_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        #        results_file_writer.writerow([formatDate_hour_min_sec,episode,successes,steps,avg_pt_step,avg_pt_movement,action_server_error])
        #avg_pt_step = avg_pt_movement = time_episode_all_step = time_episode_robot_cam = spend_time_step = spend_time_movement_step = 0

        steps = episode_reward = 0
        done = False
        reward = 0 #mg 2604
        lstm_state = None
        camera_obs, proprio_obs = env.reset()
        print("Starten der neuen Episode mit Taste 'N', aktuelle Episode: ", episode) # ANPASSUNG gmeiner 07.02
        print("Aktuelle successrate" , successes , "/" , episode ,"Episoden" )
        keyboard_obs.wait_new_episode() # ANPASSUNG gmeiner 07.02
        env.set_reward_zero()
        env.set_done_false()
        keyboard_obs.reset()

        while not done and steps < 600:  # i.e. 15 seconds
            start_time = time.time()
            start_step_time = time.process_time()
            ret, frame = cap.read()#

            cv2.namedWindow("realsenseD435i", cv2.WINDOW_NORMAL) #prevent automated resizing of the window
            cv2.resizeWindow("realsenseD435i", 640, 960) #resize window to specified dimension
            cv2.moveWindow("realsenseD435i", 1300, 100) #move window
            frame = cv2.resize(frame, (256,256)) #width, height
            cv2.imshow("realsenseD435i", frame) #show window
            if cv2.waitKey(1) == ord('p'): #
                break #
            action, lstm_state = policy.predict(camera_obs, proprio_obs, lstm_state)
            print("Episodenschritt x von 400:", steps)#
            #print("lstm_state", lstm_state)#
            #print("proprio_obs", proprio_obs)#
            print("action in evaluate:", action)#
            start_robot_cam_time = time.process_time()
            next_camera_obs, next_proprio_obs, reward, done = env.step(action, cap) #
            spend_time_movement_step = time.process_time() - start_robot_cam_time
            camera_obs, proprio_obs = next_camera_obs, next_proprio_obs
            episode_reward += reward
            print("Episode_reward:", episode_reward)
            steps += 1
            #loop_sleep(start_time)

            if keyboard_obs.reset_button:
                env.resetRoboPos()
                camera_obs, proprio_obs = env.reset()
                print("Starten der neuen Episode mit Taste 'N/Start', aktuelle Episode: ", episode)
                keyboard_obs.wait_new_episode()
                env.set_reward_zero()
                keyboard_obs.reset()
                steps = episode_reward = 0 # MG 2604
                lstm_state = None # MG 2604
                #hier noch steps zuruecksetzen und episode reward auf null
            
            #if keyboard_obs.evaluate_success:
            #    env.set_done_true()
            #ine179    env.set_reward_true()

            #if keyboard_obs.evaluate_failure:
            #    env.set_done_true()
            #    env.set_reward_zero()

            if keyboard_obs.episode_reached_button: # ANPASSUNG gmeiner 07.02
                env.set_done_true()
                #done = True # ANPASSUNG Gmeiner 07.02 wichtig feur reward
                env.set_reward_true()
            else:
                loop_sleep(start_time)

            if keyboard_obs.reset_move_client_button:
                #env.move_client_kuka.connect()
                keyboard_obs.reset_move_client_button = False
                env.resetRoboPos()
                camera_obs, proprio_obs = env.reset()
                print("Starten der neuen Episode mit Taste 'N/Start', aktuelle Episode: ", episode)
                keyboard_obs.wait_new_episode()
                env.set_reward_zero()
                keyboard_obs.reset()
                steps = episode_reward = 0 # MG 2604
                lstm_state = None # MG 2604
                action_server_robot_instance.disconnect()
                error += 1

            if keyboard_obs.success_button:
                env.set_done_true()
                env.set_reward_true()

            if keyboard_obs.failure_button:
                env.set_done_true()
                env.set_reward_zero()

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
        with open('process_information_evaluate.csv', mode='a',newline='') as results_file:
                results_file_writer = csv.writer(results_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                results_file_writer.writerow([formatDate_hour_min_sec,episode,successes,steps,avg_pt_step,avg_pt_movement,action_server_error])
        avg_pt_step = avg_pt_movement = time_episode_all_step = time_episode_robot_cam = spend_time_step = spend_time_movement_step = 0
        
        if episode_reward > 0:
            successes += 1
        wandb.log({"reward": episode_reward, "episode": episode})


        if keyboard_obs.stop_program_client_button:
            print("stop")
            #env.resetRoboPos()
            break
        
    success_rate = successes / episodes
    wandb.run.summary["success_rate"] = success_rate
    print(f'episodes {episode}')
    print(f'successess {successes}')
    print(f'success rate {success_rate}')
    print("Anzahl Actionserver Fehler:", action_server_error)
    return


def main(config):
    try:
        policy = Policy(config).to(device)
        print("fail here")
        model_path = "data/" + config["task"] + "/" + config["feedback_type"] + "_policy.pt"
        print(model_path)
        try:
            policy.load_state_dict(torch.load(model_path))
        except:
            print("can not load")
        print(model_path)

        policy.eval()
        env = CustomEnv(config)
        robot_instance = env.get_move_client_instance()
        csv_header = ['time','episode', 'amount_success','amount_steps','average_step_process_time','average_movement_process_time', 'action_server_errors',]
            #if not os.path.exists(self.image_directory+'/'+self.formatDate_direction): # create a directory to sort the result per day
            #    os.makedirs(self.image_directory+'/'+self.formatDate_direction)
        if not os.path.exists('/home/faps/CEILing/CEILing256_v2/data/'+ config["task"] + '/process_information_evaluate.csv'): # create a directory to sort the results of process
                os.chdir('/home/faps/CEILing/CEILing256_v2/data/' + config["task"])
                with open('process_information_evaluate.csv', mode='a',newline='') as results_file:
                    results_file_writer = csv.writer(results_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                    results_file_writer.writerow(csv_header)
        pass
        #keyboard_obs = KeyboardObserver()
        #cap = cv2.VideoCapture(4)
        for i in range(50): #to get a real images after a view frame messages
            ret, frame = cap.read()
            if frame is not None:
                frame = cv2.resize(frame, (256,256)) #width, height
                cv2.imwrite('startimage_evaluate.jpg',frame)
        print("Ausgangsbild fertig")
        env.image_resize_transpose(frame)
        time.sleep(5)
        run_simulation(env, policy, config["episodes"])
        cap.release()#
        cv2.destroyAllWindows()#

    except:
        print("Error")

    finally:
        robot_instance.disconnect()
    return


if __name__ == "__main__":
    set_seeds(1996)
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--feedback_type",
        dest="feedback_type",
        default="ceiling_full",
        help="options: pretraining, cloning_1, cloning_2, cloning_4, cloning_6, cloning_10, cloning_100, cloning_200, evaluative, dagger, iwr, ceiling_full, ceiling_partial",
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
        "episodes": 50, #
        "static_env": False,
        "headless_env": False,
        "proprio_dim": 8,
        "action_dim": 7,
        "visual_embedding_dim": 256,
        "learning_rate": 3e-4,
        "weight_decay": 3e-6,
        "batch_size": 8,#16
    }
    wandb.init(config=config_defaults, project="ceiling_eval", mode="online")
    config = wandb.config  # important, in case the sweep gives different values
    main(config)