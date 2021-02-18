import numpy as np
from collections import namedtuple


class SumTree(object):
    def __init__(self, capacity):
        self.data_pointer = 0
        self.capacity = capacity

        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.experience = namedtuple(
            "Experience",
            field_names=["obs", "action", "reward", "next_obs", "done"],
        )

    def add(self, priority, obs, action, reward, next_obs, done):
        tree_index = self.data_pointer + self.capacity - 1
        exp = self.experience(obs, action, reward, next_obs, done)
        self.data[self.data_pointer] = exp

        self.update(tree_index, priority)

        self.data_pointer += 1

        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]
