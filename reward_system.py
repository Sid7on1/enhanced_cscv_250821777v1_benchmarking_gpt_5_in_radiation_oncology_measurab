import logging
import numpy as np
from typing import Dict, List, Tuple
from reward_system.config import Config
from reward_system.exceptions import RewardSystemError
from reward_system.models import RewardModel
from reward_system.utils import calculate_velocity, calculate_flow_theory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardSystem:
    """
    Reward calculation and shaping system.

    This class is responsible for calculating rewards based on the agent's actions and the environment's state.
    It uses the velocity-threshold and Flow Theory algorithms to calculate rewards.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward system.

        Args:
            config (Config): Configuration object containing reward system settings.
        """
        self.config = config
        self.reward_model = RewardModel(config)

    def calculate_reward(self, action: np.ndarray, state: np.ndarray) -> float:
        """
        Calculate the reward for the given action and state.

        Args:
            action (np.ndarray): Action taken by the agent.
            state (np.ndarray): Current state of the environment.

        Returns:
            float: Calculated reward.
        """
        try:
            # Calculate velocity
            velocity = calculate_velocity(action, state)

            # Calculate Flow Theory reward
            flow_theory_reward = calculate_flow_theory(velocity, self.config.velocity_threshold)

            # Calculate reward using the reward model
            reward = self.reward_model.calculate_reward(flow_theory_reward, state)

            return reward

        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

    def shape_reward(self, reward: float) -> float:
        """
        Shape the reward to fit the agent's learning curve.

        Args:
            reward (float): Reward to be shaped.

        Returns:
            float: Shaped reward.
        """
        try:
            # Apply reward shaping using the reward model
            shaped_reward = self.reward_model.shape_reward(reward)

            return shaped_reward

        except RewardSystemError as e:
            logger.error(f"Error shaping reward: {e}")
            return 0.0

class RewardModel:
    """
    Reward model used for calculating and shaping rewards.

    This class uses a simple linear model to calculate rewards and apply shaping.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward model.

        Args:
            config (Config): Configuration object containing reward model settings.
        """
        self.config = config

    def calculate_reward(self, flow_theory_reward: float, state: np.ndarray) -> float:
        """
        Calculate the reward using the reward model.

        Args:
            flow_theory_reward (float): Reward calculated using Flow Theory.
            state (np.ndarray): Current state of the environment.

        Returns:
            float: Calculated reward.
        """
        try:
            # Calculate reward using a simple linear model
            reward = flow_theory_reward + self.config.linear_model_coefficient * np.sum(state)

            return reward

        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

    def shape_reward(self, reward: float) -> float:
        """
        Shape the reward to fit the agent's learning curve.

        Args:
            reward (float): Reward to be shaped.

        Returns:
            float: Shaped reward.
        """
        try:
            # Apply reward shaping using a simple linear model
            shaped_reward = reward + self.config.shaping_coefficient * reward

            return shaped_reward

        except RewardSystemError as e:
            logger.error(f"Error shaping reward: {e}")
            return 0.0

class Config:
    """
    Configuration object for the reward system.

    This class contains settings for the reward system, including the velocity threshold and reward model coefficients.
    """

    def __init__(self):
        """
        Initialize the configuration object.
        """
        self.velocity_threshold = 0.5
        self.linear_model_coefficient = 0.1
        self.shaping_coefficient = 0.2

class RewardSystemError(Exception):
    """
    Exception raised for reward system errors.
    """

    def __init__(self, message: str):
        """
        Initialize the exception.

        Args:
            message (str): Error message.
        """
        self.message = message

def calculate_velocity(action: np.ndarray, state: np.ndarray) -> float:
    """
    Calculate the velocity using the given action and state.

    Args:
        action (np.ndarray): Action taken by the agent.
        state (np.ndarray): Current state of the environment.

    Returns:
        float: Calculated velocity.
    """
    try:
        # Calculate velocity using a simple formula
        velocity = np.linalg.norm(action) / np.linalg.norm(state)

        return velocity

    except RewardSystemError as e:
        logger.error(f"Error calculating velocity: {e}")
        return 0.0

def calculate_flow_theory(velocity: float, velocity_threshold: float) -> float:
    """
    Calculate the Flow Theory reward using the given velocity and velocity threshold.

    Args:
        velocity (float): Calculated velocity.
        velocity_threshold (float): Velocity threshold.

    Returns:
        float: Calculated Flow Theory reward.
    """
    try:
        # Calculate Flow Theory reward using a simple formula
        flow_theory_reward = 1.0 if velocity > velocity_threshold else 0.0

        return flow_theory_reward

    except RewardSystemError as e:
        logger.error(f"Error calculating Flow Theory reward: {e}")
        return 0.0