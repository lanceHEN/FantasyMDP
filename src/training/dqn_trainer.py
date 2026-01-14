import random

import torch
import torch.nn as nn

from model import FantasyDQN
from envs import FantasyDraftEnv
from utils import POS, POS_TO_IDX

class FantasyDQNTrainer:
    """
    The FantasyDQNTrainer class provides a simple interface to train a DQN
    network for the fantasy football draft. It takes in a FantasyDQN model
    and trains it via gradient descent to find a good drafting policy, given
    who has been drafted.
    
    Attributes:
        env (FantasyDraftEnv): The environment to train from.
        model (FantasyDQN): Model to train.
        criterion (nn.Module): Loss to use for G.D.
        optimizer (torch.optim.Optimizer) Optimizer to use for G.D.
        replay_buffer (List[Tuple[np.array, int, float, np.array]]): List of
            (current_state, action, reward, next_state) tuples to sample
            minibatches from for model training.
        epsilon (float): Initial probability of moving to a random state.
        decay_rate (float): Rate at which epsilon decays at each training step.
    """
    
    def __init__(self, env: FantasyDraftEnv, model: FantasyDQN, criterion: nn.Module,
                 optimizer: torch.optim.Optimizer, epsilon: float = 1, decay_rate: float = 0.9995):
        """
        Initializes a FantasyDQNTrainer with the given environment, model,
        criterion, optimizer, and epsilon and decay rate for epsilon-greedy
        exploration.
        
        env (FantasyDraftEnv): The environment to train from.
        model (FantasyDQN): Model to train.
        criterion (nn.Module): Loss to use for G.D.
        optimizer (torch.optim.Optimizer) Optimizer to use for G.D.
        epsilon (float): Initial probability of moving to a random state.
        decay_rate (float): Rate at which epsilon decays at each training step.
        """
        self.env = env
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.replay_buffer = []
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        
    def _train_backbone(self, steps: int, warm_up: bool, minibatch_size: int = 32):
        """
        Provides a central source for common exploration code between the
        warm-up and training phases--which should differ only in whether to run
        gradient descent and use epsilon-greedy or not. This will take the
        given number of steps through the environment and adds the (state,
        action, reward, next_state) tuple to the replay buffer. If warm_up
        is False this will take optimal actions with probability epsilon (which
        will decay) and run gradient descent after each step with the given
        minibatch size.
        
        steps (int): The number of steps to take.
        warm_up (bool): Whether the model is warming up or not.
        minibatch_size (int): Number of tuples to sample from the replay buffer
            and run S.G.D. from--only useful if not warming up.
        """
        cur_state = self.env.reset()
        
        for _ in range(steps):
            # Take a step, recording the state, action, reward, and next state
            if warm_up:
                action = random.sample(self.env.available_pos())
            else:
                action_idx = torch.argmax(self.model(cur_state), dim=-1)
                action = POS(action_idx)
                
            next_state, reward, done, _ = self.env.step(action)
            
            # Add to replay buffer
            self.replay_buffer.append((cur_state, POS_TO_IDX[action], reward, next_state))
            
            # If done, need to start a new draft
            if done:
                cur_state = self.env.reset()
            else:
                cur_state = next_state
                
            if warm_up:
                self.epsilon *= self.decay_rate
                
                minibatch = random.sample(self.replay_buffer, minibatch_size)
                
                preds = torch.vstack([self.model(s[0])[s[1]] for s in minibatch])
                targets = torch.vstack([s[2] + self.model(s[3])  for s in minibatch])
                
                self.optimizer.zero_grad()    
                loss = self.criterion(preds, targets)
                loss.backward()
                self.optimizer.step()

        
    def warm_up(self, steps: int = 10000) -> None:
        """
        Takes the given number of random steps through the environment and adds
        the (state, action, reward, next_state) tuple to the replay buffer.
        
        Note one should warm up training before immediately starting epsilon-
        greedy (the train step).
        
        steps (int): The number of steps to take during the warm-up phase.
        """
        self._train_backbone(steps, True)
            
        
    def train(self, steps: int = 100000, minibatch_size: int = 32) -> None:
        """
        Trains the DQN model, taking one step at a time following epsilon-greedy,
        adding the results to the replay buffer, sampling a minibatch of the
        given size from the replay buffer, running S.G.D. with that minibatch
        and updating the DQN, and decaying epsilon.
        
        Note it is expected one has already warmed up the model by now.
        
        Also note one may call this function multiple times, since we keep
        track of epsilon.
        
        steps (int): Number of steps to train.
        minibatch_size (int): Number of tuples to sample from the replay buffer
            and run S.G.D. from.
        """
        self._train_backbone(steps, False, minibatch_size)