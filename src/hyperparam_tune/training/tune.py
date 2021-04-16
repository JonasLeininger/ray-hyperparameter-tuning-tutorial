import ray
import ray.tune as tune
from hyperopt import hp
from ray.tune import CLIReporter
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler

from hyperparam_tune.config.config import Config
from hyperparam_tune.training.trainer import Trainer


def main():
    config = Config(config_file="src/hyperparam_tune/config/config_local.yaml").config

    ray.init(address="auto")

    space = {
        "batch_size": hp.choice("batch_size", [16, 32]),
        "learning_rate": hp.loguniform("learning_rate", 0.0001, 0.01)
    }

    reporter = CLIReporter()
    reporter.add_metric_column("loss")
    reporter.add_metric_column("accuracy")
    reporter.add_metric_column("val_loss")
    reporter.add_metric_column("val_accuracy")

    ahb = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="val_accuracy",
        mode="max",
        grace_period=10,
        max_t=100)

    tune.run(
        Trainer,
        name="cifar10-asynHyper-tutorial",
        scheduler=ahb,
        config=config,
        queue_trials=True,
        num_samples=20,
        progress_reporter=reporter,
        resources_per_trial={"cpu": 3, "gpu": 0.5},
        search_alg=HyperOptSearch(
            space=space, max_concurrent=2, metric="val_accuracy", mode="max"
        ),
        checkpoint_freq=2,
        checkpoint_at_end=True,
        verbose=1,
    )


if __name__ == "__main__":
    main()