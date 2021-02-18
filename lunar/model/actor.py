import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPGActor(nn.Module):
	def __init__(self, config):
		super(DDPGActor, self).__init__()
		self.linear1 = nn.Linear(config["env_obs_dim"], 400)
		self.linear2 = nn.Linear(400, 300)
		self.linear3 = nn.Linear(300, config["action_dim"])
		self.to(config["device"])

	def forward(self, x):
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = torch.tanh(self.linear3(x))
		return x