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
from ultralytics import YOLO  # Import YOLOv8
import torch.nn.functional as F
import numpy as np

cap = cv2.VideoCapture(4)  # Capture video from camera index 4

def detect_stator(camera_batch, yolo_model):
    # Input batch shape is (600, 1, 3, 256, 256) - sequence preserved
    #print(f"Original camera_batch shape: {camera_batch.shape}")
    # Reshape camera_batch from (600, 1, 3, 256, 256) to (600, 3, 256, 256)
    b_size, f_size, _, _, _ = camera_batch.shape
    batch_tensor = camera_batch.view(-1, 3, 256, 256)
    #print(f"Reshaped batch_tensor shape: {batch_tensor.shape}")

    # Calculate padding for resizing to 640x640 while keeping the aspect ratio centered
    target_size = 640
    padding = (target_size - 256) // 2
    #print(f"Padding applied to each side: {padding}")

    # Pad the images to resize them to 640x640 with padding value set to 255 (white)
    batch_tensor_resized = F.pad(batch_tensor, (padding, padding, padding, padding), mode='constant', value=255)
    #print(f"Resized batch_tensor shape after padding: {batch_tensor_resized.shape}")
    #print(f"Range of pixel values in batch_tensor_resized after padding: Min={batch_tensor_resized.min().item()}, Max={batch_tensor_resized.max().item()}")

    # Convert to numpy for YOLO processing
    batch_numpy = batch_tensor_resized.permute(0, 2, 3, 1).cpu().numpy()  # Convert to [600, 640, 640, 3]
    batch_numpy = batch_numpy.astype(np.uint8)  # Ensure the images are in uint8 for OpenCV
    #print(f"Converted batch to numpy shape for YOLO processing: {batch_numpy.shape}")
    #print(f"Image dtype: {batch_numpy[0].dtype}, Range: {batch_numpy[0].min()} - {batch_numpy[0].max()}")

    #save_limit = 100  # Set a limit for saving images
    assert batch_numpy[0].dtype == np.uint8, "Image not in uint8 format"



    # Save the first few original images for debugging
#    for i, image_np in enumerate(batch_numpy):
#        if i < save_limit:
#            image_path = os.path.join(original_dir, f"original_image_{i}.png")
#            cv2.imwrite(image_path, image_np)
#            print(f'Saved original image to: {image_path}')

    # YOLO expects images in a list form
    batch_images = [image for image in batch_numpy]

    # Run inference on the entire batch of images
    try:
        results = yolo_model(batch_images)  # Feed the reshaped batch to the model
        #print("Ran YOLO detection on the batch.")
    except Exception as e:
        print(f"Error during YOLO inference: {e}")
        return None, []

    all_boxes = []  # Store detected bounding boxes

    # Inside the loop where bounding boxes are drawn
    for i, result in enumerate(results):
        num_boxes = len(result.boxes) if hasattr(result, 'boxes') else 0
        #print(f"Number of boxes detected in image {i}: {num_boxes}")
        # Make a copy of the image to avoid any reference issues with OpenCV
        image_np = batch_numpy[i].copy()
        #print(f"Image {i} shape: {image_np.shape}, dtype: {image_np.dtype}")

        # Draw the bounding box with the highest confidence if there are any detected
        if num_boxes > 0:
            # Get the box with the highest confidence
            best_box = max(result.boxes, key=lambda box: box.conf[0])
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])  # Get bounding box coordinates as integers
        #    print(f"Drawing box with coordinates: {(x1, y1, x2, y2)} on image {i}")
            

            try:
                # Draw the best bounding box on the image
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 5)

                # Optionally, put confidence text
                if hasattr(best_box, 'conf'):
                    conf = best_box.conf[0]
                    cv2.putText(image_np, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #            print(f"Confidence for best box: {conf:.2f}")

                # Store the best box coordinates for further use
                all_boxes.append((x1, y1, x2, y2, conf.item()))

            except cv2.error as e:
                print(f"OpenCV error when drawing rectangle on image {i}: {e}")
                continue  # Skip to the next image if an error occurs

        else:
            print(f"No boxes detected for image {i}")
        batch_numpy[i] = image_np
        # Save the modified image with bounding boxes


    # Convert numpy batch back to a tensor and resize it to 256x256
    batch_numpy_resized_back = np.array(batch_numpy)
    batch_numpy_resized_back = batch_numpy_resized_back[:, padding:-padding, padding:-padding, :]
    #print(f"Resized numpy batch back to original dimensions: {batch_numpy_resized_back.shape}")
    #for i in range(100):
    #        detected_image_path = os.path.join(detected_dir, f"detected_image_{i}.png")
    #        cv2.imwrite(detected_image_path, batch_numpy_resized_back[i])
    #        print(f'Saved detected image with bounding boxes to: {detected_image_path}')

    # Convert back to tensor and scale pixel values from [0, 255] to [0, 1]
    batch_tensor_converted = torch.tensor(batch_numpy_resized_back).permute(0, 3, 1, 2).float()
    #print(f"Converted batch back to tensor shape: {batch_tensor_converted.shape}")

    # Reshape the tensor batch back to its original shape [600, 1, 3, 256, 256]
    reshaped_batch = batch_tensor_converted.view(b_size, f_size, 3, 256, 256)
    #print(f"Reshaped batch back to original shape: {reshaped_batch.shape}")
    # Set a limit for saving images (for example, save the first 5 images)
    save_limits = 50

    # Convert the reshaped tensor (reshaped_batch) back to numpy and scale it to [0, 255]
    reshaped_numpy = reshaped_batch.view(-1, 3, 256, 256).permute(0, 2, 3, 1).cpu().numpy()   # Convert to [600, 256, 256, 3]
    reshaped_numpy = reshaped_numpy.astype(np.uint8)  # Ensure the images are in uint8 format
    print(f"Converted reshaped_batch back to numpy shape: {reshaped_numpy.shape}")

    # Save the first few images after conversion back to the original dimensions
    #for i in range(min(save_limits, reshaped_numpy.shape[0])):  # Save up to the save_limit
     #   image_path = os.path.join(original_dir, f"reshaped_image_{i}.png")
      #  cv2.imwrite(image_path, reshaped_numpy[i])
       # print(f"Saved reshaped image to: {image_path}")
    
    return reshaped_batch



def run_simulation(env, policy, episodes, yolo_model):
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
        episode_reward = 0  # Reward accumulated in the current episode
        reward = 0
        done = False  # Whether the episode has ended
        lstm_state = None  # Used to track LSTM states if policy has one

        # Reset environment for a new episode
        camera_obs, proprio_obs = env.reset()
        print(f"Start the new episode: {episode}")
        print(f"Current success rate: {successes}/{episode} Episodes")
        keyboard_obs.wait_new_episode()  # Wait for user input to start the episode
        env.set_reward_zero()
        env.set_done_false()
        keyboard_obs.reset()

        # Loop through steps in the episode until it's done or step limit is reached
        while not done and steps < 800:  # larger than average sequence_len of episode used to train
            start_time = time.time()

            # Capture camera frame
            ret, frame = cap.read()
            cv2.imshow("realsenseD435i", cv2.resize(frame, (256, 256)))  # Show window
            if cv2.waitKey(1) == ord('p'):  # Pause simulation if 'p' is pressed
                break

            # Prepare the image as a batch and detect objects using YOLO
            camera_batch = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float() / 255.0
            detected_camera_batch, _ = detect_stator(camera_batch, yolo_model)

            # Get the action from the policy model using detected images
            action, lstm_state = policy.predict(detected_camera_batch.squeeze(1), proprio_obs, lstm_state)
            print(f"Episode step: {steps}")
            print(f"Action: {action}")

            # Interact with the environment using the predicted action
            next_camera_obs, next_proprio_obs, reward, done = env.step(action, cap)

            # Update observations and accumulate reward
            camera_obs, proprio_obs = next_camera_obs, next_proprio_obs
            episode_reward += reward
            steps += 1

            # Handle keyboard input to reset, stop, or log actions
            if keyboard_obs.reset_button:
                env.resetRoboPos()
                camera_obs, proprio_obs = env.reset()
                print(f"Reset episode {episode}")
                keyboard_obs.wait_new_episode()
                env.set_reward_zero()
                keyboard_obs.reset()
                steps = episode_reward = 0
                lstm_state = None

            if keyboard_obs.episode_reached_button:
                env.set_done_true()
                env.set_reward_true()

            loop_sleep(start_time)

        # Track successful episodes
        if episode_reward > 0:
            successes += 1
        wandb.log({"reward": episode_reward, "episode": episode})

        if keyboard_obs.stop_program_client_button:
            print("Stopping the program.")
            break

    # Calculate and log success rate
    success_rate = successes / episodes
    wandb.run.summary["success_rate"] = success_rate
    print(f'Episodes: {episodes}, Successes: {successes}, Success rate: {success_rate}')

def main(config):
    """
    Main function that loads the model, initializes the environment, and runs the simulation.
    """
    try:
        policy = Policy(config).to(device)
        model_path = f"/media/faps/48508964-42c5-4ee6-a752-b91d87c4e30a/CEILing/CEILing256_v2/data/{config['task']}/{config['feedback_type']}_policy.pt"
        print(f"Loading model from {model_path}")

        policy.load_state_dict(torch.load(model_path))  # Load pre-trained policy
        policy.eval()  # Set policy model to evaluation mode

        env = CustomEnv(config)  # Initialize custom environment

        # Initialize the YOLO model for object detection
        yolo_model = YOLO('best.pt')  # Path to the trained YOLO model

        # Capture initial image from the camera
        for _ in range(50):  # Capture a few initial frames
            ret, frame = cap.read()
            if frame is not None:
                frame = cv2.resize(frame, (256, 256))
                cv2.imwrite('startimage_evaluate.jpg', frame)
        print("Initial image captured")
        env.image_resize_transpose(frame)
        time.sleep(5)

        # Run the simulation
        run_simulation(env, policy, config["episodes"], yolo_model)

        cap.release()  # Release the video capture object
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

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
        "batch_size": 4,
    }

    wandb.init(config=config_defaults, project="ceiling_eval", mode="disabled")
    config = wandb.config
    main(config)