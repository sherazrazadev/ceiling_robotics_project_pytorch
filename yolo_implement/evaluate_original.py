import torch
import wandb
import time
from models import Policy
from custom_env import CustomEnv
from utils import device, loop_sleep, set_seeds
from utils import KeyboardObserver
from argparse import ArgumentParser
import cv2
import csv
import os
import datetime

cap = cv2.VideoCapture(4)  # Capture video from camera index 4

def run_simulation(env, policy, episodes):
    """
    Runs the simulation for the specified number of episodes, interacting with the environment using a policy.
    """
    successes = 0  # Count of successful episodes
    steps = 0
    action_server_error = 0  # Track the number of action server errors
    avg_pt_step = avg_pt_movement = time_episode_all_step = time_episode_robot_cam = spend_time_step = spend_time_movement_step = 0
    keyboard_obs = KeyboardObserver()  # Capture keyboard input events
    time.sleep(10)  # Give time to initialize

    for episode in range(episodes):
        steps = 0  # Steps per episode
        episode_reward = 0 # Reward accumulated in the current episode
        reward = 0  
        done = False  # Whether the episode has ended
        lstm_state = None  # Used to track LSTM states if policy has one
        
        # Reset environment for a new episode
        camera_obs, proprio_obs = env.reset()
        print("Start the new episode with button ‘N’, current episode: ", episode) # ANPASSUNG gmeiner 07.02
        print("Current success rate" , successes , "/" , episode ,"Episodes" )
        keyboard_obs.wait_new_episode()  # Wait for user input to start the episode
        env.set_reward_zero()
        env.set_done_false()
        keyboard_obs.reset()

        # Loop through steps in the episode until it's done or step limit is reached
        while not done and steps < 800:  # larger than average sequence_len of episode used to train
            start_time = time.time()
            start_step_time = time.process_time()

            # Capture camera frame
            ret, frame = cap.read()
            cv2.namedWindow("realsenseD435i", cv2.WINDOW_NORMAL) #prevent automated resizing of the window
            cv2.resizeWindow("realsenseD435i", 640, 960) #resize window to specified dimension
            cv2.moveWindow("realsenseD435i", 1300, 100) #move window
            frame = cv2.resize(frame, (256,256)) #width, height
            cv2.imshow("realsenseD435i", frame) #show window
            if cv2.waitKey(1) == ord('p'):  # Pause simulation if 'p' is pressed
                break

            # Get the action from the policy model using current observations
            action, lstm_state = policy.predict(camera_obs, proprio_obs, lstm_state)
            print("Episode step x of 600:", steps)#
            print("Action in evaluate:", action)

            # Interact with the environment using the predicted action
            start_robot_cam_time = time.process_time()
            next_camera_obs, next_proprio_obs, reward, done = env.step(action, cap)
            spend_time_movement_step = time.process_time() - start_robot_cam_time

            # Update observations and accumulate reward
            camera_obs, proprio_obs = next_camera_obs, next_proprio_obs
            episode_reward += reward
            print("Episode_reward:", episode_reward)
            steps += 1

            # Handle keyboard input to reset, stop, or log actions
            if keyboard_obs.reset_button:
                env.resetRoboPos()
                camera_obs, proprio_obs = env.reset()
                print("Start the new episode with button ‘N’, current episode: ", episode)
                keyboard_obs.wait_new_episode()
                env.set_reward_zero()
                keyboard_obs.reset()
                steps = episode_reward = 0
                lstm_state = None

            if keyboard_obs.episode_reached_button:
                env.set_done_true()
                env.set_reward_true()
            else:
                loop_sleep(start_time)

            if keyboard_obs.reset_move_client_button:
                keyboard_obs.reset_move_client_button = False
                env.resetRoboPos()
                camera_obs, proprio_obs = env.reset()
                print("Start the new episode with button ‘N’, current episode: ", episode)
                keyboard_obs.wait_new_episode()
                env.set_reward_zero()
                keyboard_obs.reset()
                steps = episode_reward = 0 
                lstm_state = None 
                action_server_error += 1

            if keyboard_obs.success_button:
                env.set_done_true()
                env.set_reward_true()

            if keyboard_obs.failure_button:
                env.set_done_true()
                env.set_reward_zero()

            if keyboard_obs.stop_program_client_button:
                print("Program stopped.")
                env.resetRoboPos()
                break

            # Time tracking for performance metrics
            spend_time_step = time.process_time() - start_step_time
            time_episode_all_step += spend_time_step
            time_episode_robot_cam += spend_time_movement_step

            print("Duration for one step", spend_time_step)
            print("Duration for one robot/cam step", spend_time_movement_step)
            print("Duration for one episode", time_episode_all_step)
            print("Duration for one episode robot/cam step", time_episode_robot_cam)
    
        # Calculate average times for the episode and log them
        avg_pt_step = time_episode_all_step / (steps + 1)
        avg_pt_movement = time_episode_robot_cam / (steps + 1)
        now = datetime.datetime.now()
        formatDate_hour_min_sec = now.strftime("%H_%M_%S")

        # Save episode metrics to CSV
        os.chdir('/media/faps/48508964-42c5-4ee6-a752-b91d87c4e30a/CEILing/CEILing256_v2/data/' + config["task"])
        with open('process_information_evaluate.csv', mode='a',newline='') as results_file:
                results_file_writer = csv.writer(results_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                results_file_writer.writerow([formatDate_hour_min_sec,episode,successes,steps,avg_pt_step,avg_pt_movement,action_server_error])
        avg_pt_step = avg_pt_movement = time_episode_all_step = time_episode_robot_cam = spend_time_step = spend_time_movement_step = 0
        
        # Track successful episodes
        if episode_reward > 0:
            successes += 1
        wandb.log({"reward": episode_reward, "episode": episode})

        if keyboard_obs.stop_program_client_button:
            print("stop")
            break

    # Calculate and log success rate
    success_rate = successes / episodes
    wandb.run.summary["success_rate"] = success_rate
    print(f'Episodes: {episodes}, Successes: {successes}, Success rate: {success_rate}')
    print(f"Action server errors: {action_server_error}")

def main(config):
    """
    Main function that loads the model, initializes the environment, and runs the simulation.
    """
    try:
        policy = Policy(config).to(device)
        model_path = f"/media/faps/48508964-42c5-4ee6-a752-b91d87c4e30a/CEILing/CEILing256_v2/data/{config['task']}/{config['feedback_type']}_policy.pt"
        print(f"Loading model from {model_path}")
        
        try:
            policy.load_state_dict(torch.load(model_path))  # Load pre-trained policy
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        policy.eval()  # Set policy model to evaluation mode

        env = CustomEnv(config)  # Initialize custom environment
        robot_instance = env.get_move_client_instance()

        csv_header = ['time','episode', 'amount_success','amount_steps','average_step_process_time','average_movement_process_time', 'action_server_errors',]
            #if not os.path.exists(self.image_directory+'/'+self.formatDate_direction): # create a directory to sort the result per day
            #    os.makedirs(self.image_directory+'/'+self.formatDate_direction)
        if not os.path.exists('/media/faps/48508964-42c5-4ee6-a752-b91d87c4e30a/CEILing/CEILing256_v2/data/'+ config["task"] + '/process_information_evaluate.csv'): # create a directory to sort the results of process
                os.chdir('/media/faps/48508964-42c5-4ee6-a752-b91d87c4e30a/CEILing/CEILing256_v2/data/' + config["task"])
                with open('process_information_evaluate.csv', mode='a',newline='') as results_file:
                    results_file_writer = csv.writer(results_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                    results_file_writer.writerow(csv_header)
        pass

        # Capture initial image from the camera and save it
        for _ in range(50):  # Capture a few initial frames for image stabilization
            ret, frame = cap.read()
            if frame is not None:
                frame = cv2.resize(frame, (256, 256))
                cv2.imwrite('startimage_evaluate.jpg', frame)
        print("Initial image captured")
        env.image_resize_transpose(frame)
        time.sleep(5)

        # Run the simulation
        run_simulation(env, policy, config["episodes"])
        cap.release()  # Release the video capture object
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

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
        help="options: pretraining, cloning_1, cloning_2, cloning_4, cloning_6, cloning_10, cloning_100, evaluative, dagger, iwr, ceiling_full, ceiling_partial"
    )
    parser.add_argument(
        "-t",
        "--task", 
        dest="task", 
        default="GraspStator",
        help="options: ApproachCableConnector, ApproachCableStrand, ApproachStator, GraspCableConnector, GraspStator, PickUpStatorReal"
    )
    args = parser.parse_args()
    
    config_defaults = {
        "feedback_type": args.feedback_type,
        "task": args.task,
        "episodes": 51, 
        "static_env": False,
        "headless_env": False,
        "proprio_dim": 8,
        "action_dim": 7,
        "visual_embedding_dim": 256,
        "learning_rate": 3e-4,
        "weight_decay": 3e-6,
        "batch_size": 1,
    }
    
    wandb.init(config=config_defaults, project="ceiling_eval", mode="disabled")
    config = wandb.config
    main(config)