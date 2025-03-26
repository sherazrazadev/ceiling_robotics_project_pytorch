import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

# Add calculate_metrics function here
def calculate_metrics(predictions, action_batch):
    # Check if predictions have more than one dimension
    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=1).cpu().numpy()
    else:
        predictions = torch.argmax(predictions, dim=0).cpu().numpy()

    action_batch = action_batch.cpu().numpy()
    
    # Ensure metrics handle cases with no positive predictions or actions
    accuracy = np.mean(predictions == action_batch)
    
    true_positive = np.sum((predictions == action_batch) & (action_batch == 1))
    predicted_positive = np.sum(predictions == 1)
    actual_positive = np.sum(action_batch == 1)

    precision = true_positive / (predicted_positive + 1e-8)  # Avoid division by zero
    recall = true_positive / (actual_positive + 1e-8)        # Avoid division by zero
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # Add small epsilon to avoid division by zero

    return accuracy, precision, recall, f1


class Policy(nn.Module):
    def __init__(self, config, rank, world_size):
        super(Policy, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank % world_size}')
        else:
            self.device = torch.device('cpu')
        
        # Define convolutional layers with batch normalization and max-pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(512)
        
        # Calculate the LSTM input dimension based on conv output
        self.conv_output_size = 512 * 1 * 1  # Adjusted assuming input image size is 128x128 and with pooling
        lstm_dim = self.conv_output_size + config["proprio_dim"]

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=lstm_dim, hidden_size=lstm_dim, batch_first=True)
        
        # Define output layer
        self.linear_out = nn.Linear(lstm_dim, config["action_dim"])
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=config["learning_rate"], 
            weight_decay=config["weight_decay"]
        )
        
        # Standard deviation for action distribution
        self.std = 0.1 * torch.ones(config["action_dim"], dtype=torch.float32).to(self.device)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.4)  # Increased dropout rate

    def forward_step(self, camera_obs, proprio_obs, lstm_state):
        # Pass through convolutional layers with batch normalization and pooling
        vis_encoding = F.elu(self.bn1(self.conv1(camera_obs)))
        vis_encoding = self.pool1(vis_encoding)
        
        vis_encoding = F.elu(self.bn2(self.conv2(vis_encoding)))
        vis_encoding = self.pool2(vis_encoding)
        
        vis_encoding = F.elu(self.bn3(self.conv3(vis_encoding)))
        
        vis_encoding = F.elu(self.bn4(self.conv4(vis_encoding)))
        vis_encoding = self.pool3(vis_encoding)

        vis_encoding = F.elu(self.bn5(self.conv5(vis_encoding)))
        
        vis_encoding = F.elu(self.bn6(self.conv6(vis_encoding)))

        # Flatten the output from the convolutional layers
        vis_encoding = torch.flatten(vis_encoding, start_dim=1)
        
        # Concatenate with proprioception input and add an extra dimension for LSTM
        low_dim_input = torch.cat((vis_encoding, proprio_obs), dim=-1).unsqueeze(0)
        
        # Pass through LSTM layer
        lstm_out, lstm_state = self.lstm(low_dim_input, lstm_state)
        lstm_out = self.dropout(lstm_out)  # Apply dropout to LSTM output
        
        # Pass through output layer
        out = torch.tanh(self.linear_out(lstm_out))
        
        return out, lstm_state


    def forward(self, camera_obs_traj, proprio_obs_traj, action_traj, feedback_traj):
        losses = []
        lstm_state = None
        for idx in range(len(proprio_obs_traj)):
            mu, lstm_state = self.forward_step(
                camera_obs_traj[idx], proprio_obs_traj[idx], lstm_state
            )
            distribution = Normal(mu, self.std)
            log_prob = distribution.log_prob(action_traj[idx])
            loss = -log_prob * feedback_traj[idx]
            losses.append(loss)
        total_loss = torch.cat(losses).mean()
        return total_loss

    def update_params(
        self, camera_obs_traj, proprio_obs_traj, action_traj, feedback_traj, rank
    ):
        camera_obs = camera_obs_traj.to(self.device)
        proprio_obs = proprio_obs_traj.to(self.device)
        action = action_traj.to(self.device)
        feedback = feedback_traj.to(self.device)
        self.optimizer.zero_grad()
        loss = self.forward(camera_obs, proprio_obs, action, feedback)
        loss.backward()
        self.optimizer.step()
        
        # Calculate metrics after updating parameters
        with torch.no_grad():
            predictions, _ = self.forward_step(camera_obs[-1], proprio_obs[-1], None)
            accuracy, precision, recall, f1 = calculate_metrics(predictions, action[-1])
        
        training_metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        return training_metrics

    def predict(self, camera_obs, proprio_obs, lstm_state):
        camera_obs_th = torch.tensor(camera_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_th, lstm_state = self.forward_step(camera_obs_th, proprio_obs_th, lstm_state)
            action = action_th.detach().cpu().squeeze(0).numpy()
            action[-1] = binary_gripper(action[-1])
        return action, lstm_state

def binary_gripper(gripper_action):
    return 0.9 if gripper_action >= 0 else -0.9