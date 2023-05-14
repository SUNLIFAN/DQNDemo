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
            action = np.random.choice(np.arange(self.args.n_actions))
            # print("============action shape: " + str(action.shape))
        else:
            inputs = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                inputs = inputs.cuda()
            q_value = self.policy.q_network(inputs).detach()
            # print("============q_value shape: " + str(q_value.shape))
            action = torch.argmax(q_value)
            action = action.cpu().numpy()
            # print("=============action again: " + str(action.shape))
        return action.copy()

    def learn(self, transitions, logger):
        self.policy.train(transitions, logger)