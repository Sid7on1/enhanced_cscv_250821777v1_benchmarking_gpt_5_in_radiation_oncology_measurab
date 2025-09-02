import logging
import numpy as np
import torch
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from collections import deque
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MEMORY_SIZE = 100000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99

# Enum for memory types
class MemoryType(Enum):
    EXPERIENCE = 1
    TRANSITION = 2

# Dataclass for experience
@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

# Dataclass for transition
@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

# Abstract base class for memory
class Memory(ABC):
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.lock = Lock()

    @abstractmethod
    def add(self, experience: Experience):
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[Experience]:
        pass

# Class for experience replay memory
class ExperienceReplayMemory(Memory):
    def __init__(self, memory_size: int):
        super().__init__(memory_size)

    def add(self, experience: Experience):
        with self.lock:
            self.memory.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        with self.lock:
            batch = np.random.choice(len(self.memory), batch_size, replace=False)
            return [self.memory[i] for i in batch]

# Class for transition memory
class TransitionMemory(Memory):
    def __init__(self, memory_size: int):
        super().__init__(memory_size)

    def add(self, transition: Transition):
        with self.lock:
            self.memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        with self.lock:
            batch = np.random.choice(len(self.memory), batch_size, replace=False)
            return [self.memory[i] for i in batch]

# Class for experience replay buffer
class ExperienceReplayBuffer:
    def __init__(self, memory_size: int):
        self.memory = ExperienceReplayMemory(memory_size)
        self.lock = Lock()

    def add(self, experience: Experience):
        with self.lock:
            self.memory.add(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        with self.lock:
            return self.memory.sample(batch_size)

# Class for transition buffer
class TransitionBuffer:
    def __init__(self, memory_size: int):
        self.memory = TransitionMemory(memory_size)
        self.lock = Lock()

    def add(self, transition: Transition):
        with self.lock:
            self.memory.add(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        with self.lock:
            return self.memory.sample(batch_size)

# Class for experience replay agent
class ExperienceReplayAgent:
    def __init__(self, memory_size: int, batch_size: int, learning_rate: float, gamma: float):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory = ExperienceReplayBuffer(memory_size)
        self.transition_buffer = TransitionBuffer(memory_size)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def add_experience(self, experience: Experience):
        self.memory.add(experience)

    def add_transition(self, transition: Transition):
        self.transition_buffer.add(transition)

    def sample_experience(self) -> List[Experience]:
        return self.memory.sample(self.batch_size)

    def sample_transition(self) -> List[Transition]:
        return self.transition_buffer.sample(self.batch_size)

    def train(self):
        experiences = self.sample_experience()
        transitions = self.sample_transition()

        states = torch.tensor([e.state for e in experiences], dtype=torch.float32)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        next_states = torch.tensor([e.next_state for e in experiences], dtype=torch.float32)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.bool)

        state_values = self.model(states)
        next_state_values = self.model(next_states)

        q_values = state_values.gather(1, actions.unsqueeze(1))
        next_q_values = next_state_values.max(1)[0]

        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = (q_values - expected_q_values.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update transition buffer
        for i, experience in enumerate(experiences):
            transition = Transition(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                next_state=next_states[i],
                done=dones[i]
            )
            self.add_transition(transition)

    def get_state_value(self, state: np.ndarray) -> float:
        return self.model(torch.tensor(state, dtype=torch.float32)).item()

# Usage
if __name__ == "__main__":
    agent = ExperienceReplayAgent(MEMORY_SIZE, BATCH_SIZE, LEARNING_RATE, GAMMA)

    # Add experiences
    agent.add_experience(Experience(np.array([1, 2, 3, 4]), 0, 10, np.array([5, 6, 7, 8]), False))
    agent.add_experience(Experience(np.array([9, 10, 11, 12]), 1, 20, np.array([13, 14, 15, 16]), True))

    # Train
    agent.train()

    # Get state value
    state = np.array([17, 18, 19, 20])
    state_value = agent.get_state_value(state)
    print(state_value)