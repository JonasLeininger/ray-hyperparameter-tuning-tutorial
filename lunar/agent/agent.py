import numpy as np
import torch
import torch.nn.functional as F

from lunar.agent.ornstein_uhlenbeck_noise import Noise
from lunar.model.actor import DDPGActor
from lunar.model.critic import DDPGCritic

class Agent():

    def __init__(self, config):
        self.actor_local = DDPGActor(config)
        self.actor_target = DDPGActor(config)
        self.optimizer_actor = torch.optim.Adam(
            self.actor_local.parameters(), lr=config["learning_rate"]
        )
        self.critic_local = DDPGCritic(config)
        self.critic_target = DDPGCritic(config)
        self.optimizer_critic = torch.optim.Adam(
            self.critic_local.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["optimizer_critic_weight_decay"],
        )
        self.hard_update(self.critic_local, self.critic_target)
        self.hard_update(self.actor_local, self.actor_target)

        self.noise = Noise(
            config["action_dim"],
            np.asarray(config["noise_mu"]),
            theta=config["noise_theta"],
            sigma=config["noise_sigma"],
        )
        self.gamma = config["agent_gamma"]
        self.tau = config["agent_tau"]
        self.device = config["device"]

    def act(self, observation, add_noise=True):
        obs = torch.tensor(observation, dtype=torch.float, device=self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(obs)
            action = action.cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
                noise = self.noise.sample()
                action += noise
        return np.clip(action, -1, 1)
    
    def learn(self, experience, weights=None, weighted_loss=False):
        states, actions, rewards, next_states, dones = experience
        target_actions = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, target_actions)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = self.critic_local(states, actions)
        delta = torch.abs(q_expected - q_targets)

        if not weighted_loss:
            self.critic_loss = F.mse_loss(q_expected, q_targets)
        else:
            weights = torch.from_numpy(weights).float().to(self.device)
            loss = F.mse_loss(q_expected, q_targets)
            w_loss = loss * weights
            self.critic_loss = w_loss.mean()

        self.optimizer_critic.zero_grad()
        self.critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.optimizer_critic.step()

        actions_pred = self.actor_local(states)
        self.actor_loss = -self.critic_local(states, actions_pred).mean()

        self.optimizer_actor.zero_grad()
        self.actor_loss.backward()
        self.optimizer_actor.step()

        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

        return delta.clone().cpu().data.numpy()
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def save_checkpoint_for_tune(self, checkpoint_path):
        torch.save(
            {
                "actor_model_state_dict": self.actor_target.state_dict(),
                "actor_optimizer_state_dict": self.optimizer_actor.state_dict(),
                "critic_model_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.optimizer_critic.state_dict(),
            },
            checkpoint_path,
        )

    def load_checkpoint_for_tune(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.actor_target.load_state_dict(checkpoint["actor_model_state_dict"])
        self.optimizer_actor.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.actor_local.load_state_dict(checkpoint["actor_model_state_dict"])
        self.optimizer_actor.load_state_dict(checkpoint["actor_optimizer_state_dict"])

        self.critic_target.load_state_dict(checkpoint["critic_model_state_dict"])
        self.optimizer_critic.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.critic_local.load_state_dict(checkpoint["critic_model_state_dict"])
        self.optimizer_critic.load_state_dict(checkpoint["critic_optimizer_state_dict"])
