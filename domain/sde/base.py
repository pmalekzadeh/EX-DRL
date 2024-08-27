'''Abstract base class of SDE environments'''
import sys
from abc import ABC, abstractmethod

import numba
import numpy as np

from dr.domain_randomized import DomainRandomized
from patterns.observations import Observable


@numba.njit
def set_seed(value):
    np.random.seed(value)

class SDESimulator(Observable, DomainRandomized, ABC):
    """Abstract base class of SDE environments
    Simulate SDE system of Stock price and Vanilla Option Implied volatility
    
    Methods
    -------
    step(): Evolute SDE system
    stock_price(): Stock price
    implied_vol():  Implied volatility

    """
    def __init__(self, year_T=252, step_t=1, **kwargs) -> None:
        self.T = year_T
        self.dt = step_t / year_T
        Observable.__init__(self)
        DomainRandomized.__init__(self, **kwargs)
    
    
    def __len__(self):
        return sys.maxsize
    
    def has_branch(self):
        return False

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def stock_price(self):
        pass

    @abstractmethod
    def implied_vol(self, ttm, moneyness):
        pass

    @abstractmethod
    def reset(self, *args, **kwargs):
        pass

