import os
import logging
from typing import Dict, List
import torch
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration class
class Config:
    """
    Configuration class for the agent and environment.

    Parameters:
    ----------
    config_file : str
        Path to the configuration file.

    Attributes:
    ----------
    agent : Dict
        Configuration settings for the agent.
    environment : Dict
        Configuration settings for the environment.
    """
    def __init__(self, config_file: str):
        # Read configuration file
        if not os.path.exists(config_file):
            raise ValueError(f"Configuration file '{config_file}' not found.")
        self.config = self._read_config(config_file)

        # Set default values for mandatory attributes
        self.agent = {
            "learning_rate": 0.001,
            "hidden_size": 256,
            "dropout": 0.5,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.995
        }
        self.environment = {
            "state_size": (10, 10),
            "action_space": [0, 1, 2, 3],
            "reward_range": [-1, 1],
            "gamma": 0.95
        }

        # Override default values with values from the config file
        if "agent" in self.config:
            self.agent.update(self.config["agent"])
        if "environment" in self.config:
            self.environment.update(self.config["environment"])

        # Validate configuration
        self._validate_config()

        logger.info("Configuration loaded successfully.")

    def _read_config(self, config_file: str) -> Dict:
        """Read configuration file and return as a dictionary."""
        try:
            with open(config_file, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading configuration file: {e}")

    def _validate_config(self):
        """Validate the configuration and raise errors for invalid settings."""
        # Validate agent configuration
        if not isinstance(self.agent["learning_rate"], float):
            raise ValueError("Invalid learning rate. Must be a float.")
        if not isinstance(self.agent["hidden_size"], int):
            raise ValueError("Invalid hidden size. Must be an integer.")
        if not 0 <= self.agent["dropout"] <= 1:
            raise ValueError("Invalid dropout rate. Must be between 0 and 1.")
        if not isinstance(self.agent["epsilon_start"], float):
            raise ValueError("Invalid initial epsilon. Must be a float.")
        if not isinstance(self.agent["epsilon_end"], float):
            raise ValueError("Invalid final epsilon. Must be a float.")
        if not isinstance(self.agent["epsilon_decay"], float):
            raise ValueError("Invalid epsilon decay. Must be a float.")

        # Validate environment configuration
        if not isinstance(self.environment["state_size"], tuple) or len(self.environment["state_size"]) != 2:
            raise ValueError("Invalid state size. Must be a tuple of length 2.")
        if not isinstance(self.environment["action_space"], list) or not all(isinstance(a, int) for a in self.environment["action_space"]):
            raise ValueError("Invalid action space. Must be a list of integers.")
        if not isinstance(self.environment["reward_range"], tuple) or len(self.environment["reward_range"]) != 2:
            raise ValueError("Invalid reward range. Must be a tuple of length 2.")
        if not isinstance(self.environment["gamma"], float):
            raise ValueError("Invalid discount factor. Must be a float.")

# Algorithm-specific constants and thresholds
VELOCITY_THRESHOLD = 0.1
FLOW_THEORY_CONSTANT = 0.5

# Paper-specific metrics
class Metrics:
    """
    Class to compute and store metrics mentioned in the research paper.

    ...

    Attributes:
    ----------
    rewards : List
        List to store episode rewards.
    episode_lengths : List
        List to store episode lengths.

    Methods:
    -------
    compute_average_reward(self) -> float:
        Compute the average reward over all episodes.
    compute_average_episode_length(self) -> float:
        Compute the average episode length.
    """

    def __init__(self):
        self.rewards = []
        self.episode_lengths = []

    def add_reward(self, reward: float):
        """Add a reward to the list of episode rewards."""
        self.rewards.append(reward)

    def add_episode_length(self, length: int):
        """Add an episode length to the list of episode lengths."""
        self.episode_lengths.append(length)

    def compute_average_reward(self) -> float:
        """Compute the average reward over all episodes."""
        if not self.rewards:
            return 0.0
        return sum(self.rewards) / len(self.rewards)

    def compute_average_episode_length(self) -> float:
        """Compute the average episode length."""
        if not self.episode_lengths:
            return 0
        return sum(self.episode_lengths) / len(self.episode_lengths)

# Main function to create configuration
def create_configuration(config_file: str) -> Dict:
    """
    Create and validate the configuration for the agent and environment.

    Parameters:
    ----------
    config_file : str
        Path to the configuration file.

    Returns:
    -------
    Dict
        The validated configuration settings for the agent and environment.
    """
    try:
        # Create configuration object
        config = Config(config_file)

        # Return configuration settings
        return {
            "agent": config.agent,
            "environment": config.environment
        }
    except ValueError as e:
        logger.error(f"Error creating configuration: {e}")
        return None

# Example usage
if __name__ == "__main__":
    config_file = "path/to/config.json"
    configuration = create_configuration(config_file)
    if configuration:
        print("Agent configuration:", configuration["agent"])
        print("Environment configuration:", configuration["environment"])
    else:
        print("Error creating configuration.")