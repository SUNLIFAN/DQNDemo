import numpy as np
import torch
from algorithm.dqn import DQN

class DQN_Agent:
    def __init__(self, args):
        self.args = args
        self.policy = DQN(args)

    def select_action(self, states, epsilon):
        # TODO: 补全epsilon_greedy代码实现
        if np.random.uniform() < epsilon:
            action =
        else:
            inputs = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                inputs = inputs.cuda()
            q_value =
            action =
            action = action.cpu().numpy()
        return action.copy()

    def learn(self, transitions, logger):
        self.policy.train(transitions, logger)