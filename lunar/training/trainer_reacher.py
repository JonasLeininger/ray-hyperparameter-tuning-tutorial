import os
import numpy as np
from ray import tune

from torch.utils.tensorboard import SummaryWriter
from unityagents import UnityEnvironment

from utils.annotations import override
from config.config import Config
from agent.agent_bipedal import  BipedalAgent
from model.prioritized_experience_replay_buffer import PrioritizedExperienceReplay

class ReacherTrainer(tune.Trainable):
    _name = "ReacherTrainer"

    @override(tune.Trainable)
    def setup(self, config):
        self.env = UnityEnvironment(file_name=config["path_to_env_novis"])
        self.init_env()
        self.agent = BipedalAgent(config)
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
        self.score = 0
        self.mean_of_mean = 0
    
    def init_env(self):
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.obs = self.env_info.vector_observations
        self.state_dim = self.obs.shape[1]
        self.action_dim = self.brain.vector_action_space_size

    @override(tune.Trainable)
    def step(self):
        self.writer = SummaryWriter(log_dir=self.logdir)
        self._execute_iteration()
        self.rewards.append(self.reward)
        self._weights_for_tensorboard_log(self.train_count)
        self._reward_bookkeeping()
        
        return {"score": self.score, "mean_of_mean": self.mean_of_mean, "exp": self.experience_count, "done": self.done}
    
    def _reward_bookkeeping(self):
        self.score = np.mean(self.reward)
        self.mean_rewards.append(self.score)

        if self.train_count >= 50:
            self.mean_of_mean = np.mean(self.mean_rewards[-50:])
        else:
            self.mean_of_mean = np.mean(self.reward)

        if self.mean_of_mean >= self.config["success_threshold"]:
            self.done = True

        if self.train_count == self.config["max_training_iterations"]:
            self.done = True

    def _execute_iteration(self):
        self._reset_env()
        self._run_environment_episode()
        self.train_count += 1

    def _reset_env(self):
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.obs = self.env_info.vector_observations
        self.timestep_count_env = 0
        self.dones = np.array(0)
        self.reward = np.zeros((20, 1))

    def _run_environment_episode(self):
        while not np.any(self.dones.astype(dtype=bool)):
            self.experience_count += 1 * self.obs.shape[0]
            actions = self.agent.act(self.obs)
            self.env_info = self.env.step(actions)[self.brain_name]
            next_obs = self.env_info.vector_observations
            rewards = self.env_info.rewards
            rewards = np.array(rewards).reshape((20, 1))
            dones = np.array(self.env_info.local_done).reshape((20, 1))
            self.reward = np.add(self.reward, rewards)

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
    config = Config(config_file="config/config_local.yaml").config
    trainer = ReacherTrainer(config=config)
    for i in range(10):
        result_dict = trainer.step()
        print(result_dict)