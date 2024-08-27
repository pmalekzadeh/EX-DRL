'''SABR Model'''
from typing import Union

import numpy as np
import numba

from .base import SDESimulator
from dr.generator import DRGenerator
from config.config_loader import ConfigLoader


@ConfigLoader.register_class
class SABRSimulator(SDESimulator):
    def __init__(self, 
                 s: Union[float, DRGenerator], 
                 vol: Union[float, DRGenerator],  
                 volvol: Union[float, DRGenerator]=0.6, 
                 beta=1, rho=-0.7, 
                 year_T=252, step_t=1):
        self.mu = self.r = self.q = 0.0
        super().__init__(year_T=year_T, step_t=step_t, 
                         s=s, vol=vol, volvol=volvol,
                         beta=beta, rho=rho)
        initial_values = self.initialize()
        self.s = initial_values['s']
        self.vol = initial_values['vol']
        self.beta = initial_values['beta']
        self.rho = initial_values['rho']
        self.volvol = initial_values['volvol']
        

    @staticmethod
    @numba.jit(nopython=True)    
    def _sabr_sim(s, vol, mu, dt, beta, rho, volvol):
        """Simulate SABR model
        1). stock price
        2). instantaneous vol
        Returns:
            np.ndarray: stock price in shape (num_path, num_period)
            np.ndarray: instantaneous vol in shape (num_path, num_period)
        """
        qs = np.random.normal()
        qi = np.random.normal()
        qv = rho * qs + np.sqrt(1 - rho * rho) * qi
        
        gvol = vol * (s ** (beta - 1))
        s_next = s * np.exp(
            (mu - (gvol ** 2) / 2) * dt +
            gvol * np.sqrt(dt) * qs
        )
        vol_next = vol * np.exp(
            -volvol * volvol * 0.5 * dt +
            volvol * qv * np.sqrt(dt)
        )
        return s_next, vol_next
    
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _sabr_implied_vol(s, vol, volvol, r, q, beta, rho, ttm, moneyness):
        """Convert SABR instantaneous vol to option implied vol

        Args:
            ttm (np.ndarray): time to maturity in shape (num_period,)
            moneyness (np.ndarray): moneyness = K / S in shape (num_period,) 

        Returns:
            np.ndarray: implied vol in shape (num_path, num_period)
        """
        K = moneyness * s
        F = s * np.exp((r - q) * ttm)
        x = (F * K) ** ((1 - beta) / 2)
        y = (1 - beta) * np.log(F / K)
        A = vol / (x * (1 + y * y / 24 + y * y * y * y / 1920))
        B = 1 + ttm * (
            ((1 - beta) ** 2) * (vol * vol) / (24 * x * x)
            + rho * beta * volvol * vol / (4 * x)
            + volvol * volvol * (2 - 3 * rho * rho) / 24
        )
        Phi = (volvol * x / vol) * np.log(F / K)
        Chi = np.log((np.sqrt(1 - 2 * rho * Phi + Phi *
                     Phi) + Phi - rho) / (1 - rho))
        sbar_iv = np.zeros_like(ttm)
        for i, (f, k, a, b, phi, chi) in enumerate(zip(F, K, A, B, Phi, Chi)):
            if f == k:
                sbar_iv[i] = vol * b / (f ** (1 - beta))
            else:
                sbar_iv[i] = a * b * phi / (chi if chi != 0 else 1e-6)
        return sbar_iv


    def step(self):
        self.s, self.vol = self._sabr_sim(self.s, self.vol, self.mu, self.dt, self.beta, self.rho, self.volvol)
        self.notify_observers({'action': 'step', 's': self.s, 'vol': self.vol})

    def stock_price(self):
        return self.s
    
    def implied_vol(self, ttm, moneyness):
        return self._sabr_implied_vol(self.s, self.vol, self.volvol, self.r, self.q, self.beta, 
                                      self.rho, ttm/self.T, moneyness)

    def reset(self):
        initial_values = self.initialize()
        self.s = initial_values['s']
        self.vol = initial_values['vol']
        self.beta = initial_values['beta']
        self.rho = initial_values['rho']
        self.volvol = initial_values['volvol']
        self.notify_observers({'action': 'reset', 's': self.s, 'vol': self.vol})