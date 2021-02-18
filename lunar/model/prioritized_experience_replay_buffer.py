from collections import namedtuple

import numpy as np
import torch

from lunar.model.sum_tree import SumTree


class PrioritizedExperienceReplay(object):
    # stored as ( s, a, r, s_ ) in SumTree

    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.0

    def __init__(self, capacity):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.experience = namedtuple(
            "Experience",
            field_names=["obs", "action", "reward", "next_obs", "done"],
        )

    def _getPriority(self, error):
        return (error + self.PER_e) ** self.PER_a

    def store(self, sample):
        max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        if max_priority == 0.0:
            max_priority = self.absolute_error_upper

        for i in range(sample[0].shape[0]):
            obs = sample[0][i]
            action = sample[1][i]
            reward = sample[2][i]
            next_obs = sample[3][i]
            done = sample[4][i]
            self.tree.add(max_priority, obs, action, reward, next_obs, done)

    def sample(self, batch_size):
        memory_batch = []
        b_idx, b_ISWeights = (
            np.empty((batch_size,), dtype=np.int32),
            np.empty((batch_size, 1), dtype=np.float32),
        )
        priority_segment = self.tree.total_priority / batch_size

        self.PER_b = np.min([1.0, self.PER_b + self.PER_b_increment_per_sampling])
        # p_min = (np.min(self.tree.tree[-self.tree.capacity:]) + self.PER_e) / self.tree.total_priority
        # max_weight = (p_min * batch_size)**(-self.PER_b)

        # if max_weight == 0.:
        #     max_weight = 0.01

        for i in range(batch_size):
            lower_segment, higher_segment = (
                priority_segment * i,
                priority_segment * (i + 1),
            )
            value = np.random.uniform(lower_segment, higher_segment)

            index, priority, data = self.tree.get_leaf(value)
            if type(data) == int:
                data = self.tree.data[0]
                priority = self.tree.tree[self.capacity - 1]

            sampling_probability = priority / self.tree.total_priority
            if sampling_probability == 0.0:
                sampling_probability = 0.1

            b_ISWeights[i, 0] = np.power(batch_size * sampling_probability, -self.PER_b)
            b_idx[i] = index
            memory_batch.append(data)

        b_ISWeights = b_ISWeights / b_ISWeights.max(axis=0)
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.transform_to_tensor(memory_batch)
        experiences = (states, actions, rewards, next_states, dones)
        return b_idx, experiences, b_ISWeights

    def batch_updates(self, tree_idx, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)

        ps = np.power(clipped_errors, self.PER_a)

        for index, prio in zip(tree_idx, ps):
            self.tree.update(index, prio)

    def transform_to_tensor(self, experiences):
        obs = (
            torch.from_numpy(
                np.vstack(
                    np.expand_dims(
                        [e.obs for e in experiences if e is not None], axis=0
                    )
                )
            )
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack(
                    np.expand_dims(
                        [e.action for e in experiences if e is not None], axis=0
                    )
                )
            )
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack(
                    np.expand_dims(
                        [e.reward for e in experiences if e is not None], axis=0
                    )
                )
            )
            .float()
            .to(self.device)
        )
        next_obs = (
            torch.from_numpy(
                np.vstack(
                    np.expand_dims(
                        [e.next_obs for e in experiences if e is not None], axis=0
                    )
                )
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack(
                    np.expand_dims(
                        [e.done for e in experiences if e is not None], axis=0
                    )
                ).astype(np.uint8)
            )
            .float()
            .to(self.device)
        )

        return obs, actions, rewards, next_obs, dones
