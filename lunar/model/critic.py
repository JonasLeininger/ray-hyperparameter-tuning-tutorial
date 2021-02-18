import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPGCritic(nn.Module):
    def __init__(self, config):
        super(DDPGCritic, self).__init__()
        self.linear1 = nn.Linear(config["env_obs_dim"], 400)
        self.linear2 = nn.Linear(400 + config["action_dim"], 300)
        self.linear3 = nn.Linear(300, 1)
        self.to(config["device"])
    
    def forward(self, obs, actions):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(torch.cat([x, actions], 1)))
        x = self.linear3(x)
        return x
