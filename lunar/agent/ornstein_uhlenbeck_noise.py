import copy

import numpy as np


class Noise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0.0, theta=0.2, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [np.random.randn() for i in range(len(x))]
        )
        self.state = x + dx
        return self.state

    def reduce(self):
        if self.theta >= 0.01:
            self.theta -= 0.05
        if self.sigma >= 0.01:
            self.sigma -= 0.05
