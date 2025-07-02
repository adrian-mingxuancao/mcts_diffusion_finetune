"""
Core framework for MCTS/Imitation Learning-guided finetuning of protein language models.
Modular design for extensibility to new tasks, models, and reward functions.
"""

class TaskModule:
    """Base class for downstream tasks (e.g., inverse folding, representation learning)."""
    def __init__(self, config):
        self.config = config
    def get_initial_state(self, input_data):
        raise NotImplementedError
    def is_terminal(self, state):
        raise NotImplementedError
    def available_actions(self, state):
        raise NotImplementedError
    def apply_action(self, state, action):
        raise NotImplementedError

class ExpertPlanner:
    """Base class for expert/planning module (e.g., MCTS)."""
    def __init__(self, task_module, reward_module):
        self.task = task_module
        self.reward = reward_module
    def plan(self, initial_state):
        """Return expert rollout(s) for imitation learning."""
        raise NotImplementedError

class RewardModule:
    """Base class for reward/value computation (task-specific)."""
    def __init__(self, config):
        self.config = config
    def compute(self, state, action, next_state):
        raise NotImplementedError

class LearningModule:
    """Base class for learning/finetuning (imitation learning, RL, etc.)."""
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    def update(self, expert_rollouts):
        raise NotImplementedError

# Example: Registry for easy extensibility
TASK_REGISTRY = {}
PLANNER_REGISTRY = {}
REWARD_REGISTRY = {}
LEARNING_REGISTRY = {}

def register_task(name):
    def decorator(cls):
        TASK_REGISTRY[name] = cls
        return cls
    return decorator

def register_planner(name):
    def decorator(cls):
        PLANNER_REGISTRY[name] = cls
        return cls
    return decorator

def register_reward(name):
    def decorator(cls):
        REWARD_REGISTRY[name] = cls
        return cls
    return decorator

def register_learning(name):
    def decorator(cls):
        LEARNING_REGISTRY[name] = cls
        return cls
    return decorator 