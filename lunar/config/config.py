import yaml

from ray.rllib.agents.trainer import COMMON_CONFIG


class Config:
    def __init__(self, config_file="config/config_local.yaml"):
        config = self.load_config_file(config_file=config_file)
        common_c = COMMON_CONFIG
        common_c.update(config)
        self.config = common_c

    def load_config_file(self, config_file: str = "config/config_local.yaml"):
        with open(config_file, "r") as stream:
            try:
                return yaml.full_load(stream)
            except yaml.YAMLError as ye:
                print(ye)
