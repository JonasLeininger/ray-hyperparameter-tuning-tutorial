import numpy as np
from ray import tune

from lunar.utils.annotations import override


class SimpleTrainer(tune.Trainable):
    _name = "SimpleTrainer"

    @override(tune.Trainable)
    def setup(self, config):
        self.theta = config["theta"]
        self.h = config["h"]
        self.alpha = config["alpha"]
        self.id = config["trainer_id"]
        self.use_logger = False
        self.objective = lambda theta: 1.2 - np.sum(theta**2)
        self.surrogateObjective = lambda theta, h: 1.2 - np.sum(h * theta**2)
    
    @override(tune.Trainable)
    def step(self):
        div_surrogateObjective = -2.0 * self.h * self.theta
        div_loss = -div_surrogateObjective # !gradient ascent!

        self.theta -= div_loss * self.alpha
        score = self.objective(self.theta)
        return {"id": self.id, "score": score, "h": self.h, "theta": self.theta}
    
    def explore(self):
        epsilon = np.random.randn(*self.h.shape) * 0.1
        self.h += epsilon
    
    def exploit(self, best_trainer_id, params):
        if best_trainer_id != self.id:
            self.theta = np.copy(params)
            return True
        return False
