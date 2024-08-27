import unittest
import numpy as np

from domain.sde import *
from domain.asset.portfolio import Portfolio
from config.config_loader import ConfigLoader

class PortfolioTestCase(unittest.TestCase):
    def setUp(self):
        # Create a mock SDESimulator instance for testing
        config_data = {
            "test_portfolio": {
                "class_name": "Portfolio",
                "params": {
                    "sde": {
                        "ref": "sabr_sde",
                    },
                    "client_trade_poisson_rate": 0.0,
                    "client_options": {
                        "class_name": "BarrierDIPOption",
                        "params": {
                            "sde": {
                                "ref": "sabr_sde",
                            },
                            "call": [True],
                            "moneyness": [1.0],
                            "ttm": [40],
                            "shares": [1.0],
                            "barrier": [10.0],
                            # "sim_moneyness_mean": 1.0,
                            # "sim_moneyness_std": 0.0,
                            # "sim_ttms": [40],
                            # "sim_call": [True],
                            # "sim_barrier_uniform": [0.6, 0.8]
                        }
                    },
                    "hedging_options": {
                        "class_name": "VanillaOption",
                        "params": {
                            "sde": {
                                "ref": "sabr_sde",
                            },
                            "sim_moneyness_mean": 1.0,
                            "sim_moneyness_std": 0.0,
                            "sim_ttms": [20],
                            "sim_call": [True]
                        }
                    }
                }
            },
            "bsm_sde": {
                "class_name": "BSMSimulator",
                "params": {
                    "s": 10.6,
                    "vol": 0.2,
                }
            },
            "sabr_sde": {
                "class_name": "SABRSimulator",
                "params": {
                    "s": 10.6,
                    "vol": 0.2,
                    "volvol": 0.6,     
                }
            }
        }
        np.random.seed(0)
        config_loader = ConfigLoader(config_data)
        config_loader.load_objects()
        self.portfolio: Portfolio = config_loader["test_portfolio"]
        self.sde: SDESimulator = self.portfolio.sde

    def test_trade_hedging(self):
        episode_pnls = []
        n_episode = 2000
        episode_len = 20
        for i in range(n_episode):
            self.sde.reset()
            pnl = 0
            for j in range(episode_len):
                # delta hedging
                self.portfolio.trade_hedging()
                self.sde.step()
                pnl += self.portfolio.pnl
                # print(i, j, self.sde.s, pnl)
            episode_pnls.append(pnl)
        print(np.mean(episode_pnls), np.std(episode_pnls))

if __name__ == '__main__':
    unittest.main()