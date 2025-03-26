from pickle import FALSE
import random
import numpy as np
import wandb
import time
import threading #GM AE 152
import torch
from torch.utils.data import Dataset
from pynput import keyboard
from functools import partial
from collections import deque

#from rlbench.tasks import (
#    CloseMicrowave,
#    PushButton,
#    TakeLidOffSaucepan,
#    UnplugCharger,
#    PickUpStatorReal,
#) #rausgenommen funktioniert, aber erstmal auskommentiert auch bei custom_env


#task_switch = {
#    "CloseMicrowave": CloseMicrowave,
#    "PushButton": PushButton,
#    "TakeLidOffSaucepan": TakeLidOffSaucepan,
#    "UnplugCharger": UnplugCharger,
#    "PickUpStatorReal": PickUpStatorReal,
#}

#device = torch.device(f"cuda:{rank}")


result_available = threading.Event() # ANPASSUNG gmeiner 07.02

class KeyboardObserver:
    def __init__(self):
        self.last_value_gripper = None 
        self.reset()
        self.hotkeys = keyboard.GlobalHotKeys(
            {
                "g": partial(self.set_label, 1),  # good
                "b": partial(self.set_label, 0),  # bad
                "c": partial(self.set_gripper, -0.9),  # close
                "v": partial(self.set_gripper, 0.9),  # open
                "f": partial(self.set_gripper, None),  # gripper free
                "x": self.reset_episode,
                "n": self.press_key_new_episode, # neu um die Episode fuer reale Exp. neu zu starten; ANPASSUNG gmeiner 07.02
                "m": self.episode_end_reached,  # neu um die Episode manuell zu beenden; ANPASSUNG gmeiner 07.02
                "y": self.reset_move_client,
                "t": self.evaluate_success,
                "z": self.evaluate_failure,
                "ü": self.stop_program,
            }
        )
        self.hotkeys.start()
        self.direction = np.array([0, 0, 0, 0, 0, 0])
        self.listener = keyboard.Listener(
            on_press=self.set_direction, on_release=self.reset_direction
        )
        self.key_mapping = {
            #"d": (1, 1),   # left(0, 0.9, 0) | Roboter Welkoordinaten Y-
            #"a": (1, -1),  # right(0, -0.9, 0) | Roboter Welkoordinaten Y+
            #"s": (0, 1),  # backward (0.9, 0, 0) | Roboter Welkoordinaten X+
            #"w": (0, -1),  # forward (-0.9, 0, 0) | Roboter Welkoordinaten X-
            #"e": (2, 1),  # down (0,0,0.9) oben | Roboter Welkoordinaten Z+
            #"q": (2, -1),  # up (0,0,-0,9) unten | Roboter Welkoordinaten Z-
            #"j": (3, -1),  # look left | Roboter Welkoordinaten B - 
            #"l": (3, 1),  # look right | Roboter Welkoordinaten B +
            #"k": (4, -1),  # look up | Roboter Welkoordinaten C-
            #"i": (4, 1),  # look down | Roboter Welkoordinaten C+
            #"o": (5, -1),  # rotate left | Roboter Welkoordinaten A+
            #"u": (5, 1),  # rotate right | Roboter Welkoordinaten A-

            "s": (1, 1),   # left #fuer die richtige Orientierung fuer unseren Demonstrator muss das [s] werden, zuvor a ********s
            "w": (1, -1),  # right #fuer die richtige Orientierung fuer unseren Demonstrator muss das [w] werden, zuvor d 
            "a": (0, 1),  # backward #fuer die richtige Orientierung fuer unseren Demonstrator muss das [a] werden, zuvor s ********a
            "d": (0, -1),  # forward #fuer die richtige Orientierung fuer unseren Demonstrator muss das [d] werden, zuvor w
            "q": (2, 1),  # down
            "e": (2, -1),  # up #down? E+A & E+S funktioniert nicht in kombination
            "j": (3, -1),  # look left #fuer die richtige Orientierung fuer unseren Demonstrator muss das [i] werden, zuvor j 
            "l": (3, 1),  # look right #fuer die richtige Orientierung fuer unseren Demonstrator muss das [k] werden, zuvor l
            "k": (4, -1),  # look up #fuer die richtige Orientierung fuer unseren Demonstrator muss das [j] werden, zuvor i
            "i": (4, 1),  # look down #fuer die richtige Orientierung fuer unseren Demonstrator muss das [l] werden, zuvor k
            "o": (5, -1),  # rotate left
            "u": (5, 1),  # rotate right
        }
        self.listener.start()
        return

    def set_label(self, value):
        self.label = value
        print("label set to: ", value)
        return

    def get_label(self):
        return self.label

    def set_gripper(self, value):
        self.gripper_open = value
        print("gripper set to: ", value)
        return

    def get_gripper(self):
        return self.gripper_open

    def set_direction(self, key):
        #print("SET direction key:", key)
        try:
            idx, value = self.key_mapping[key.char]
            self.direction[idx] = value
        except (KeyError, AttributeError):
            #print("Fehler bei set_direction")
            pass
        return

    def reset_direction(self, key):
        #print("RESET direction key:", key)
        try:
            idx, _ = self.key_mapping[key.char]
            self.direction[idx] = 0
        except (KeyError, AttributeError):
            #print("Fehler bei set_direction")
            pass
        return

    def has_joints_cor(self):
        return self.direction.any()

    def has_gripper_update(self):        
        if self.get_gripper() == self.last_value_gripper:  #      
            response = False

        else:       
            response = True    
        self.last_value_gripper = self.get_gripper()
        return response
        

    def get_ee_action(self):
        return self.direction * 0.9

    def has_gripper_update_feedbacktrain(self):
        return self.get_gripper() is not None

    def reset_episode(self):
        print("Used episode resetting button (reject record)")
        self.reset_button = True
        return


    def begin_new_episode(self): # ANPASSUNG Gmeiner 07.02
        result_available.set()
        return

    def press_key_new_episode(self):
        if self.new_episode_button == False:
            self.new_episode_button = True
            try:
                self.thread.start()
            except (KeyError, AttributeError):
                print("Error bei press_key_new_Episode")
        return

            
        
    #def press_key_new_episode(self): # ANPASSUNG Gmeiner 07.02
    #    self.new_episode_button = True
    #    try:
    #    	self.thread.start()
    #    except (KeyError, AttributeError):
    #    	print("Error bei press_key_new_episode")
    #    return

    def wait_new_episode(self): # ANPASSUNG Gmeiner 07.02
        print("new episode 1")
        self.thread = threading.Thread(target=self.begin_new_episode)
        print(self.thread)
        print("new episode 2")
        result_available.wait(25)
        print("new episode 3")
        result_available.clear()
        print("new episode 4")
        print("resetbutton", self.reset_button)
        print("episode_reached_button", self.episode_reached_button)
        print("new_episode_button", self.new_episode_button)
        print("reset_move_client_button", self.reset_move_client_button)
        self.reset()
        print("reset conducted")
        print("resetbutton DANACH", self.reset_button)
        print("episode_reached_button DANACH", self.episode_reached_button)
        print("new_episode_button DANACH", self.new_episode_button)
        print("reset_move_client_button DANACH", self.reset_move_client_button)
        return
    
    def episode_end_reached(self): # ANPASSUNG Gmeiner 07.02
        print("Used episode ending button")
        self.episode_reached_button = True
        return 

    def evaluate_success(self):
        print("Reports manipulation success")
        self.success_button = True
        return

    def evaluate_failure(self):
        print("Report manipulation failure")
        self.failure_button = True

    def reset(self):
        self.set_label(1)
        self.set_gripper(0.9) #gripper open ->0.9 zuvor none
        self.reset_button = False
        self.episode_reached_button = False # Gmeiner
        self.new_episode_button = False # Gmeiner
        self.reset_move_client_button = False
        self.success_button = False
        self.failure_button = False
        self.stop_program_client_button = False
        return

    def reset_move_client(self): # ANPASSUNG Gmeiner 07.02
        print("Used move client resetting button")
        self.reset_move_client_button = True
        return 
    
    def stop_program(self): # ANPASSUNG Gmeiner 07.02
        print("Stop program")
        self.stop_program_client_button = True
        return 


class MetricsLogger:
    def __init__(self):
        self.total_successes = 0
        self.total_episodes = 0
        self.total_steps = 0
        self.total_cor_steps = 0
        self.total_pos_steps = 0
        self.total_neg_steps = 0
        self.episode_metrics = deque(maxlen=1)
        self.reset_episode()
        return

    def reset_episode(self):
        self.episode_reward = 0
        self.episode_steps = 0
        self.episode_cor_steps = 0
        self.episode_pos_steps = 0
        self.episode_neg_steps = 0
        return

    def log_step(self, reward, feedback):
        self.episode_reward += reward
        self.episode_steps += 1
        if feedback == -1:
            self.episode_cor_steps += 1
        elif feedback == 1:
            self.episode_pos_steps += 1
        elif feedback == 0:
            self.episode_neg_steps += 1
        else:
            raise NotImplementedError
        return

    def log_episode(self, current_episode):
        episode_metrics = {
            "reward": self.episode_reward,
            "ep_cor_rate": self.episode_cor_steps / self.episode_steps,
            "ep_pos_rate": self.episode_pos_steps / self.episode_steps,
            "ep_neg_rate": self.episode_neg_steps / self.episode_steps,
            "episode": current_episode,
        }
        self.append(episode_metrics)
        self.total_episodes += 1
        if self.episode_reward > 0:
            self.total_successes += 1
        self.total_steps += self.episode_steps
        self.total_cor_steps += self.episode_cor_steps
        self.total_pos_steps += self.episode_pos_steps
        self.total_neg_steps += self.episode_neg_steps
        self.reset_episode()
        return

    def log_session(self):
        success_rate = self.total_successes / self.total_episodes
        cor_rate = self.total_cor_steps / self.total_steps
        pos_rate = self.total_pos_steps / self.total_steps
        neg_rate = self.total_neg_steps / self.total_steps
        wandb.run.summary["success_rate"] = success_rate
        wandb.run.summary["total_cor_rate"] = cor_rate
        wandb.run.summary["total_pos_rate"] = pos_rate
        wandb.run.summary["total_neg_rate"] = neg_rate
        return

    def append(self, episode_metrics):
        self.episode_metrics.append(episode_metrics)
        return

    def pop(self):
        return self.episode_metrics.popleft()

    def empty(self):
        return len(self.episode_metrics) == 0


class TrajectoriesDataset(Dataset):
    def __init__(self, sequence_len):
        self.sequence_len = sequence_len
        self.camera_obs = []
        self.proprio_obs = []
        self.action = []
        self.feedback = []
        self.reset_current_traj()
        self.pos_count = 0
        self.cor_count = 0
        self.neg_count = 0
        return

    def __getitem__(self, idx):
        if self.cor_count < 10:
            alpha = 1
        else:
            alpha = (self.pos_count + self.neg_count) / self.cor_count
        weighted_feedback = [
            alpha if value == -1 else value for value in self.feedback[idx]
        ]
        weighted_feedback = torch.tensor(weighted_feedback).unsqueeze(1)
        return (
            self.camera_obs[idx],
            self.proprio_obs[idx],
            self.action[idx],
            weighted_feedback,
        )

    def __len__(self):
        return len(self.proprio_obs)

    def add(self, camera_obs, proprio_obs, action, feedback):
        self.current_camera_obs.append(camera_obs)
        self.current_proprio_obs.append(proprio_obs)
        self.current_action.append(action)
        self.current_feedback.append(feedback)
        if feedback[0] == 1:
            self.pos_count += 1
        elif feedback[0] == -1:
            self.cor_count += 1
        elif feedback[0] == 0:
            self.neg_count += 1
        return

    def save_current_traj(self):
        camera_obs = downsample_traj(self.current_camera_obs, self.sequence_len)
        proprio_obs = downsample_traj(self.current_proprio_obs, self.sequence_len)
        action = downsample_traj(self.current_action, self.sequence_len)
        feedback = downsample_traj(self.current_feedback, self.sequence_len)
        #print("camera_obs dtype",camera_obs.dtype)
        #print("proprio_obs dtype",proprio_obs.dtype)
        #print("action dtype",action.dtype)
        #print("feedback dtype",feedback.dtype)
        camera_obs_th = torch.tensor(camera_obs, dtype=torch.float32)
        proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32)
        action_th = torch.tensor(action, dtype=torch.float32)
        feedback_th = torch.tensor(feedback, dtype=torch.float32)
        self.camera_obs.append(camera_obs_th)
        self.proprio_obs.append(proprio_obs_th)
        self.action.append(action_th)
        self.feedback.append(feedback_th)
        #print("self.camera_obs",(self.camera_obs))
        #print("self.proprio_obs",(self.proprio_obs))
        #print("self.action",(self.action))
        #print("self.feedback",(self.feedback))
        #print("self.camera_obs type",type(self.camera_obs))
        #print("self.proprio_obs type",type(self.proprio_obs))
        #print("self.action type",type(self.action))
        #print("self.feedback type",type(self.feedback))
        #print("camera_obs dtype",self.camera_obs.dtype)
        #print("proprio_obs dtype",self.proprio_obs.dtype)
        #print("action dtype",self.action.dtype)
        #print("feedback dtype",self.feedback.dtype)
        self.reset_current_traj()
        return

    def reset_current_traj(self):
        self.current_camera_obs = []
        self.current_proprio_obs = []
        self.current_action = []
        self.current_feedback = []
        return
    
    def sample(self, batch_size):
            batch_size = min(batch_size, len(self))
            indeces = random.sample(range(len(self)), batch_size)
            batch = zip(*[self[i] for i in indeces])# compress these selected trajectories, from 0 to (batch_size-1), each torch.Size([150, 3, 256, 256]), torch.Size([150, 8]), torch.Size([150, 7]), torch.Size([150, 1])
            camera_batch = torch.stack(next(batch), dim=1)# stack to [150, 3, 16, 256, 256]) , here 16 is batch size
            proprio_batch = torch.stack(next(batch), dim=1)
            action_batch = torch.stack(next(batch), dim=1)
            feedback_batch = torch.stack(next(batch), dim=1)
            return camera_batch, proprio_batch, action_batch, feedback_batch


def downsample_traj(traj, target_len):
    if len(traj) == target_len:
        return traj
    elif len(traj) < target_len:
        return traj + [traj[-1]] * (target_len - len(traj))
    else:
        indeces = np.linspace(start=0, stop=len(traj) - 1, num=target_len)
        indeces = np.round(indeces).astype(int)
        return np.array([traj[i] for i in indeces])


def loop_sleep(start_time):
    dt = 0.05
    sleep_time = dt - (time.time() - start_time)
    print(sleep_time)
    if sleep_time > 0.0:
        time.sleep(sleep_time)
    return


def euler_to_quaternion(euler_angle):
    roll, pitch, yaw = euler_angle
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def set_seeds(seed):
    """Sets all seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
