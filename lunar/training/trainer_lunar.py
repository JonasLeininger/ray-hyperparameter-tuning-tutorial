import os
import numpy as np
from ray import tune
import gym
from torch.utils.tensorboard import SummaryWriter

from lunar.utils.annotations import override
from lunar.config.config import Config
from lunar.agent.agent import  Agent
from lunar.model.prioritized_experience_replay_buffer import PrioritizedExperienceReplay

class LunarTrainer(tune.Trainable):
    _name = "LunarTrainer"

    @override(tune.Trainable)
    def setup(self, config):
        self.env = gym.make('LunarLanderContinuous-v2')
        self.agent = Agent(config)
        self.obs = self.env.reset()
        self.per = PrioritizedExperienceReplay(
            capacity=config["replay_buffer_memory_size"]
            )
        self.done = False
        self.dones = np.array(0)
        self.experience_count = 0
        self.train_count = 0
        self.reward = 0
        self.rewards = []
        self.mean_rewards = []
    
    @override(tune.Trainable)
    def step(self):
        self.writer = SummaryWriter(log_dir=self.logdir)
        self._execute_iteration()
        self.rewards.append(self.reward)
        self._weights_for_tensorboard_log(self.train_count)
        self._reward_bookkeeping()
        

        return {"reward": self.reward, "mean_rewards": self.mean_rewards, "exp": self.experience_count}
    
    def _reward_bookkeeping(self):
        if self.train_count >= 50:
            self.mean_rewards = np.mean(self.rewards[-50:])
        else:
            self.mean_rewards = np.mean(self.rewards)

        if self.mean_rewards >= self.config["success_threshold"]:
            self.done = True

        if self.train_count == self.config["max_training_iterations"]:
            self.done = True

    def _execute_iteration(self):
        self._reset_env()
        self._run_environment_episode()
        self.train_count += 1

    def _reset_env(self):
        self.obs = self.env.reset()
        self.obs = np.expand_dims(self.obs, axis=0)
        self.timestep_count_env = 0
        self.dones = np.array(0)
        self.reward = 0

    def _run_environment_episode(self):
        while not np.any(self.dones.astype(dtype=bool)):
            self.experience_count += 1
            self.env.render(mode='rgb_array')
            if self.experience_count <= 20000:
                self.agent.noise.theta = 0.5
                self.agent.noise.sigma = 0.9
            elif self.experience_count <=50000:
                self.agent.noise.theta = 0.3
                self.agent.noise.sigma = 0.5
            else:
                self.agent.noise.theta = 0.05
                self.agent.noise.sigma = 0.05
            actions = self.agent.act(self.obs)
            next_obs, rewards, dones, info = self.env.step(actions.flatten())
            self.reward += rewards

            rewards = np.array((rewards,))
            rewards = np.expand_dims(rewards, axis=0)
            next_obs = np.expand_dims(next_obs, axis=0)
            dones = np.array((dones,))
            dones = np.expand_dims(dones, axis=0)
            experience = self.obs, actions, rewards, next_obs, dones
            self.per.store(experience)
            self._learn(self.timestep_count_env)
            self.dones = dones
            self.obs = next_obs
            self.timestep_count_env += 1

    def _learn(self, timestep_count):
        if (self.experience_count >= self.config["memory_learning_start"]) and (
            timestep_count % self.config["agent_learn_every_x_steps"] == 0
        ):
            self._train_model()

    def _train_model(self):
        for learn_iteration in range(self.config["agent_learn_num_iterations"]):
            b_idx, experience, b_ISWeights = self.per.sample(
                self.config["replay_buffer_batch_size"]
            )
            obs, actions, rewards,next_obs, dones = experience
            absolute_errors = self.agent.learn(
                experience, weights=b_ISWeights, weighted_loss=True
            )
            self.per.batch_updates(b_idx, absolute_errors)

    def _weights_for_tensorboard_log(self, train_count):
        for name, param in self.agent.critic_local.named_parameters():
            self.writer.add_histogram(
                "critic-local" + name, param.clone().cpu().data.numpy(), train_count
            )
        for name, param in self.agent.actor_local.named_parameters():
            self.writer.add_histogram(
                "actor-local" + name, param.clone().cpu().data.numpy(), train_count
            )
        for name, param in self.agent.critic_target.named_parameters():
            self.writer.add_histogram(
                "critic-target" + name, param.clone().cpu().data.numpy(), train_count
            )
        for name, param in self.agent.actor_target.named_parameters():
            self.writer.add_histogram(
                "actor-target" + name, param.clone().cpu().data.numpy(), train_count
            )
    
    @override(tune.Trainable)
    def _save(self, checkpoint_path):
        print(checkpoint_path)
        checkpoint_path_model_name = os.path.join(checkpoint_path, "model.pth")
        self.agent.save_checkpoint_for_tune(checkpoint_path_model_name)
        return checkpoint_path_model_name

    @override(tune.Trainable)
    def _restore(self, checkpoint_path):
        self.agent.load_checkpoint_for_tune(checkpoint_path)
        print(checkpoint_path)

if __name__ == "__main__":
    config = Config(config_file="lunar/config/config_lunar.yaml").config
    trainer = LunarTrainer(config=config)
    for i in range(10):
        result_dict = trainer.step()
        print(result_dict)