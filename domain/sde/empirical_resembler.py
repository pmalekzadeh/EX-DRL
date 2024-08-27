'''Black Scholes Merton SDE'''
from typing import Union

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import numba

from .base import SDESimulator
from dr.generator import DRGenerator
from config.config_loader import ConfigLoader


@ConfigLoader.register_class
class EmpiricalResembler(SDESimulator):
    def __init__(self,
                 filename: str,
                 episode_length: int,
                 s: Union[float, DRGenerator],
                 year_T=252, step_t=1):
        self.moneyness = np.array([0.85, 1.00, 1.15], dtype=float)
        self.maturities = np.array([20/year_T, 60/year_T], dtype=float)
        # hard coded for used volatility 
        super().__init__(year_T=year_T, step_t=step_t, s=s)
        initial_values = self.initialize()
        self.episode_index = -1
        self.step_index = 0
        self.episode_length = episode_length
        self._load_data(filename)
        self._prepare_path(initial_values['s'], self.episode_index)
    
    def __len__(self):
        return self.stock_logret.shape[0]

    def _load_data(self, filename: str):
        """Load empirical data

        Args:
            filename (str): historical data filename
            initial_s (float): starting stock price at time 0
        """
        df = pd.read_csv(filename, index_col=0)
        steps = self.episode_length + 1
        rolling_paths = [df.iloc[i:i+steps] for i in range(len(df)-steps)]
        self.stock_logret = np.array([rolling_paths[i]['ret'].values for i in range(len(rolling_paths))])   # shape (n_paths, steps)
        self.imp_vol_surface = np.array([
            [[rolling_paths[i]['1M_0.85'].values, rolling_paths[i]['1M_1.00'].values, rolling_paths[i]['1M_1.15']],
             [rolling_paths[i]['3M_0.85'].values, rolling_paths[i]['3M_1.00'].values, rolling_paths[i]['3M_1.15']]] for i in range(len(rolling_paths))])\
             .transpose(0, 3, 2, 1) # shape (n_paths, steps, n_moneyness, n_maturities)

    
    def _prepare_path(self, initial_s, episode_index):
        self.cur_stock_prices = initial_s * np.ones(self.episode_length+1, dtype=np.float32)
        for j in range(1, self.episode_length+1):
            prev_price = self.cur_stock_prices[j-1]
            step_ret = self.stock_logret[episode_index, j]
            self.cur_stock_prices[j] = np.exp(np.log(prev_price) + step_ret)
        self.cur_imp_vol_surface = self.imp_vol_surface[episode_index]
        self.s = initial_s
        self.vol = self.implied_vol(30, 1.0)[0]
        

    def step(self):
        self.step_index += 1
        self.s = self.cur_stock_prices[self.step_index]
        self.vol = self.implied_vol(30, 1.0)[0]
        self.notify_observers({'action': 'step', 's': self.s})

    def stock_price(self):
        return self.s
    
    def implied_vol(self, ttm, moneyness):
        """Interpolate implied volatility from surface

        Args:
            ttm (Union[int, np.ndarray]): time to maturity
            moneyness (Union[float, np.ndarray]): moneyness of the option
        """
        step_imp_vol_surface = self.cur_imp_vol_surface[self.step_index]
        ttm_year = ttm/self.T
        ttm_year = np.array([ttm_year] if isinstance(ttm, int) else ttm_year, dtype=np.float32)
        moneyness = np.array([moneyness] if isinstance(moneyness, float) else moneyness, dtype=np.float32)
        # first interpolate along two maturities for the moneyness
        interp_f1m = interp1d(self.moneyness, step_imp_vol_surface[:, 0], kind='linear', 
                              fill_value=(step_imp_vol_surface[0, 0], step_imp_vol_surface[-1, 0]), bounds_error=False)
        interp_f3m = interp1d(self.moneyness, step_imp_vol_surface[:, 1], kind='linear', 
                              fill_value=(step_imp_vol_surface[0, 1], step_imp_vol_surface[-1, 1]), bounds_error=False)
        vol_1m = interp_f1m(moneyness)
        vol_3m = interp_f3m(moneyness)
        total_var_1m = vol_1m**2*self.maturities[0]
        w_1m = (self.maturities[1]-ttm_year)/(self.maturities[1]-self.maturities[0])
        total_var_3m = vol_3m**2*self.maturities[1]
        w_3m = (ttm_year-self.maturities[0])/(self.maturities[1]-self.maturities[0])

        interp_vol = np.where(
            ttm_year < self.maturities[0], vol_1m, 
            np.where(ttm_year > self.maturities[1], vol_3m,
                     np.sqrt((total_var_1m*w_1m + total_var_3m*w_3m)/ttm_year)
            )
        )

        return interp_vol
                    
    def reset(self):
        initial_values = self.initialize()
        self.episode_index += 1
        self.step_index = 0
        self._prepare_path(initial_values['s'], self.episode_index)
        self.notify_observers({'action': 'reset', 's': self.s})
