import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """
    Class to calculate evaluation metrics for the agent.

    Attributes:
    ----------
    config : Dict
        Configuration dictionary containing parameters for evaluation metrics.
    """

    def __init__(self, config: Dict):
        """
        Initialize the EvaluationMetrics class.

        Parameters:
        ----------
        config : Dict
            Configuration dictionary containing parameters for evaluation metrics.
        """
        self.config = config
        self.velocity_threshold = config.get('velocity_threshold', 0.5)
        self.flow_theory_threshold = config.get('flow_theory_threshold', 0.8)

    def calculate_velocity(self, data: List[float]) -> float:
        """
        Calculate the velocity of the agent.

        Parameters:
        ----------
        data : List[float]
            List of values to calculate the velocity from.

        Returns:
        -------
        float
            Calculated velocity.
        """
        try:
            # Calculate the velocity using the formula from the paper
            velocity = np.mean(np.diff(data))
            return velocity
        except Exception as e:
            logger.error(f"Error calculating velocity: {str(e)}")
            return None

    def calculate_flow_theory(self, data: List[float]) -> float:
        """
        Calculate the flow theory metric of the agent.

        Parameters:
        ----------
        data : List[float]
            List of values to calculate the flow theory metric from.

        Returns:
        -------
        float
            Calculated flow theory metric.
        """
        try:
            # Calculate the flow theory metric using the formula from the paper
            flow_theory = np.mean(np.abs(np.diff(data)))
            return flow_theory
        except Exception as e:
            logger.error(f"Error calculating flow theory: {str(e)}")
            return None

    def evaluate_agent(self, data: List[float]) -> Dict:
        """
        Evaluate the agent using the calculated metrics.

        Parameters:
        ----------
        data : List[float]
            List of values to evaluate the agent from.

        Returns:
        -------
        Dict
            Dictionary containing the evaluation results.
        """
        try:
            # Calculate the velocity and flow theory metrics
            velocity = self.calculate_velocity(data)
            flow_theory = self.calculate_flow_theory(data)

            # Evaluate the agent based on the calculated metrics
            evaluation_results = {
                'velocity': velocity,
                'flow_theory': flow_theory,
                'velocity_threshold': self.velocity_threshold,
                'flow_theory_threshold': self.flow_theory_threshold
            }

            # Check if the agent meets the velocity and flow theory thresholds
            if velocity is not None and flow_theory is not None:
                if velocity > self.velocity_threshold and flow_theory > self.flow_theory_threshold:
                    evaluation_results['evaluation'] = 'Pass'
                else:
                    evaluation_results['evaluation'] = 'Fail'
            else:
                evaluation_results['evaluation'] = 'Invalid'

            return evaluation_results
        except Exception as e:
            logger.error(f"Error evaluating agent: {str(e)}")
            return None

class AgentEvaluator:
    """
    Class to evaluate the agent using the EvaluationMetrics class.

    Attributes:
    ----------
    evaluation_metrics : EvaluationMetrics
        Instance of the EvaluationMetrics class.
    """

    def __init__(self, config: Dict):
        """
        Initialize the AgentEvaluator class.

        Parameters:
        ----------
        config : Dict
            Configuration dictionary containing parameters for evaluation metrics.
        """
        self.evaluation_metrics = EvaluationMetrics(config)

    def evaluate(self, data: List[float]) -> Dict:
        """
        Evaluate the agent using the EvaluationMetrics class.

        Parameters:
        ----------
        data : List[float]
            List of values to evaluate the agent from.

        Returns:
        -------
        Dict
            Dictionary containing the evaluation results.
        """
        try:
            # Evaluate the agent using the EvaluationMetrics class
            evaluation_results = self.evaluation_metrics.evaluate_agent(data)
            return evaluation_results
        except Exception as e:
            logger.error(f"Error evaluating agent: {str(e)}")
            return None

def main():
    # Create a configuration dictionary
    config = {
        'velocity_threshold': 0.5,
        'flow_theory_threshold': 0.8
    }

    # Create an instance of the AgentEvaluator class
    agent_evaluator = AgentEvaluator(config)

    # Create a list of values to evaluate the agent from
    data = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Evaluate the agent
    evaluation_results = agent_evaluator.evaluate(data)

    # Print the evaluation results
    print(evaluation_results)

if __name__ == '__main__':
    main()