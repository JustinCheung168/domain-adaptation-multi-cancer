import json
import logging
import yaml

class ExperimentConfig():
    def __init__(self, config_file: str):
        with open(config_file, "r") as f:
            self.config_dict = yaml.safe_load(f)
        logging.info(f"Read config {config_file}:\n{json.dumps(self.config_dict, indent=4)}")

class TrainingConfig(ExperimentConfig):
    def __init__(self, config_file: str):
        super().__init__(config_file)

        # TODO - enforce that config meets expected format


class EvaluationConfig(ExperimentConfig):
    def __init__(self, config_file: str):
        super().__init__(config_file)

        # TODO - enforce that config meets expected format
