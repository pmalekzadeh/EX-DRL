import unittest
import numpy as np

import os
import yaml

from domain.sde import *
from dr import *
from domain.asset.portfolio import Portfolio
from env.trade_env import DREnv
from config.config_loader import ConfigLoader

dir_path = os.path.dirname(os.path.realpath(__file__))

class PortfolioTestCase(unittest.TestCase):
    def setUp(self):
        # Create a mock SDESimulator instance for testing
        with open(os.path.join(dir_path, '../run_configs/bsm_vanilla_env.yaml'), 'r') as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        np.random.seed(0)
        config_loader = ConfigLoader(config_data)
        config_loader.load_objects()
        self.env: DREnv = config_loader["env"]
        

    def test_trade_hedging(self):
        episode_pnls = []
        n_episode = 1000
        for i in range(n_episode):
            state = self.env.reset()
            pnl = 0
            terminal = False
            while not terminal:
                # Full Gamma and Delta Hedging
                if state[3] > 0:
                    state, reward, terminal, info = self.env.step(np.array([0.]))
                elif state[3] < 0:
                    state, reward, terminal, info = self.env.step(np.array([1.]))
                pnl += reward
            episode_pnls.append(pnl)
        print(np.mean(episode_pnls), np.std(episode_pnls))

if __name__ == '__main__':
    unittest.main()