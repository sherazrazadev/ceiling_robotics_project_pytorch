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





#256 with max pooling

class Policy(nn.Module):
    def __init__(self, config, rank, world_size):
        super(Policy, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank % world_size}')
        else:
            self.device = torch.device('cpu')

        # Adjusted LSTM dimension
        lstm_dim = 512 + config["proprio_dim"]

        # Convolutional Network with Batch Normalization and selected MaxPooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.bn6 = nn.BatchNorm2d(256)

        # LSTM and output layers
        self.lstm = nn.LSTM(lstm_dim, lstm_dim)

        # Linear output layer
        self.linear_out = nn.Linear(lstm_dim, config["action_dim"])

        # Optimizer with parameter groups for LSTM and others
        self.optimizer = torch.optim.Adam([
            {'params': self.conv1.parameters()},
            {'params': self.conv2.parameters()},
            {'params': self.conv3.parameters()},
            {'params': self.conv4.parameters()},
            {'params': self.conv5.parameters()},
            {'params': self.conv6.parameters()},
            {'params': self.lstm.parameters(), 'lr': config["learning_rate"] * 0.5},  # Reduced LR for LSTM
            {'params': self.linear_out.parameters()}
        ], lr=config["learning_rate"], weight_decay=config["weight_decay"])

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        self.std = 0.1 * torch.ones(config["action_dim"], dtype=torch.float32)
        self.std = self.std.to(self.device)
        self.dropout = nn.Dropout(p=0.4)

    def forward_step(self, camera_obs, proprio_obs, lstm_state):
        # Apply convolutional layers with Batch Normalization and selective MaxPooling
        x = F.elu(self.bn1(self.conv1(camera_obs)))
        x = self.pool1(x)

        x = F.elu(self.bn2(self.conv2(x)))

        x = F.elu(self.bn3(self.conv3(x)))
        x = self.pool2(x)

        x = F.elu(self.bn4(self.conv4(x)))

        x = F.elu(self.bn5(self.conv5(x)))
        x = self.pool3(x)

        x = F.elu(self.bn6(self.conv6(x)))

        # Flatten the output of the last conv layer
        vis_encoding = torch.flatten(x, start_dim=1)

        # Ensure compatibility with LSTM input size
        vis_encoding = F.elu(nn.Linear(vis_encoding.shape[1], 512).to(self.device)(vis_encoding))

        # Concatenate with proprioceptive observations
        low_dim_input = torch.cat((vis_encoding, proprio_obs), dim=-1).unsqueeze(0)
        low_dim_input = self.dropout(low_dim_input)

        # LSTM layer
        lstm_out, (h, c) = self.lstm(low_dim_input, lstm_state)
        lstm_state = (h, c)

        # Output layer with Tanh activation
        out = torch.tanh(self.linear_out(lstm_out))
        return out, lstm_state