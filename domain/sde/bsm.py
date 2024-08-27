'''Black Scholes Merton SDE'''
from typing import Union

import numpy as np
import numba

from .base import SDESimulator
from dr.generator import DRGenerator
from config.config_loader import ConfigLoader


@ConfigLoader.register_class
class BSMSimulator(SDESimulator):
    def __init__(self, 
                 s: Union[float, DRGenerator],
                 vol: Union[float, DRGenerator], 
                 year_T=252, step_t=1):
        self.mu = 0.
        super().__init__(year_T=year_T, step_t=step_t, s=s, vol=vol)
        initial_values = self.initialize()
        self.s = initial_values['s']
        self.vol = initial_values['vol']

    @staticmethod
    @numba.jit(nopython=True)
    def _brownian_sim(s, vol, mu, dt):
        z = np.random.normal()

        s_next = s * np.exp(
                (mu - (vol ** 2) / 2) * dt +
                vol * np.sqrt(dt) * z
            )
        return s_next
    
    def step(self):
        self.s = self._brownian_sim(self.s, self.vol, self.mu, self.dt)
        self.notify_observers({'action': 'step', 's': self.s})

    def stock_price(self):
        return self.s
    
    def implied_vol(self, ttm, moneyness):
        return np.array([self.vol])
    
    def reset(self):
        initial_values = self.initialize()
        self.s = initial_values['s']
        self.vol = initial_values['vol']
        self.notify_observers({'action': 'reset', 's': self.s, 'vol': self.vol})