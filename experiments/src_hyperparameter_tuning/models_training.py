import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
#from utils import device


class Policy(nn.Module):
    def __init__(self, config,rank,world_size):
        super(Policy, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank % world_size}')
        else:
            self.device = torch.device('cpu')
        lstm_dim = 4096 + config["proprio_dim"] #config["visual_embedding_dim"] -> 8192 = 16x16x32 -> 4096 (8x8x64)[current] --> 2048 (8x8x32) --> 4096 (16x16x16) --> 2048 (16x16x8)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2) # implementation of max pool did not result in better performance
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=5, padding=2, stride=2 # increasement of channels resulted in better performance 3->2->1->1 to 3->8->16->32->64->64

        )
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=5, padding=2, stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2
        )
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2
        )
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=2 # added more layers to reduce lstm dimension  (8x8x64)
        )
        self.lstm = nn.LSTM(lstm_dim, lstm_dim)  # , batch_first=True)
        self.linear_out = nn.Linear(lstm_dim, config["action_dim"])
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        self.std = 0.1 * torch.ones(config["action_dim"], dtype=torch.float32)
        self.std = self.std.to(self.device)
        self.dropout = nn.Dropout(p=0.4)
        return


    def forward_step(self, camera_obs, proprio_obs, lstm_state):
        vis_encoding = F.elu(self.conv1(camera_obs))
        vis_encoding = F.elu(self.conv2(vis_encoding))
        vis_encoding = F.elu(self.conv3(vis_encoding))
        vis_encoding = F.elu(self.conv4(vis_encoding))
        vis_encoding = F.elu(self.conv5(vis_encoding))
        vis_encoding = torch.flatten(vis_encoding, start_dim=1)
        low_dim_input = torch.cat((vis_encoding, proprio_obs), dim=-1).unsqueeze(0)
        low_dim_input = self.dropout(low_dim_input)
        lstm_out, (h, c) = self.lstm(low_dim_input, lstm_state)
        lstm_state = (h, c)
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
        self, camera_obs_traj, proprio_obs_traj, action_traj, feedback_traj,rank
    ):
        camera_obs = camera_obs_traj.to(self.device)
        proprio_obs = proprio_obs_traj.to(self.device)
        action = action_traj.to(self.device)
        feedback = feedback_traj.to(self.device)
        self.optimizer.zero_grad()
        loss = self.forward(camera_obs, proprio_obs, action, feedback)
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"Device: {rank}, Gradient of {name}: {param.grad}")
        self.optimizer.step()
        training_metrics = {"loss": loss}
        return training_metrics

    def predict(self, camera_obs, proprio_obs, lstm_state):
        camera_obs_th = torch.tensor(camera_obs, dtype=torch.float32).unsqueeze(0)
        proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32).unsqueeze(0)
        camera_obs_th = camera_obs_th.to(self.device)
        proprio_obs_th = proprio_obs_th.to(self.device)
        with torch.no_grad():
            action_th, lstm_state = self.forward_step(
                camera_obs_th, proprio_obs_th, lstm_state
            )
            action = action_th.detach().cpu().squeeze(0).squeeze(0).numpy()
            action[-1] = binary_gripper(action[-1])
        return action, lstm_state


def binary_gripper(gripper_action):
    if gripper_action >= 0.0:
        gripper_action = 0.9
    elif gripper_action < 0.0:
        gripper_action = -0.9
    return gripper_action
