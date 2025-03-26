import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class Policy(nn.Module):
    def __init__(self, config, rank, world_size):
        super(Policy, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank % world_size}')
        else:
            self.device = torch.device('cpu')

        # Adjusted LSTM dimension
        lstm_dim = 512 + config["proprio_dim"]
        
        # Convolutional Network with reduced channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)  # Reduced stride to avoid excessive size reduction
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)  # Reduced stride to avoid excessive size reduction

        # MaxPooling layer with smaller kernel size
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # LSTM and output layers
        self.lstm = nn.LSTM(lstm_dim, lstm_dim)
        self.linear_out = nn.Linear(lstm_dim, config["action_dim"])
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        self.std = 0.1 * torch.ones(config["action_dim"], dtype=torch.float32)
        self.std = self.std.to(self.device)
        self.dropout = nn.Dropout(p=0.4)

    def forward_step(self, camera_obs, proprio_obs, lstm_state):
        # Apply convolutional layers and maxpooling
        vis_encoding = F.elu(self.conv1(camera_obs))
        vis_encoding = F.elu(self.conv2(vis_encoding))
        vis_encoding = self.maxpool(vis_encoding)

        vis_encoding = F.elu(self.conv3(vis_encoding))
        vis_encoding = F.elu(self.conv4(vis_encoding))
        vis_encoding = self.maxpool(vis_encoding)

        vis_encoding = F.elu(self.conv5(vis_encoding))
        vis_encoding = F.elu(self.conv6(vis_encoding))

        # Flatten the output of the last conv layer
        vis_encoding = torch.flatten(vis_encoding, start_dim=1)

        # Ensure compatibility with LSTM input size
        vis_encoding = F.elu(nn.Linear(vis_encoding.shape[1], 512).to(self.device)(vis_encoding))
        
        low_dim_input = torch.cat((vis_encoding, proprio_obs), dim=-1).unsqueeze(0)
        low_dim_input = self.dropout(low_dim_input)
        
        lstm_out, (h, c) = self.lstm(low_dim_input, lstm_state)
        lstm_state = (h, c)
        
        out = torch.tanh(self.linear_out(lstm_out))
        return out, lstm_state
