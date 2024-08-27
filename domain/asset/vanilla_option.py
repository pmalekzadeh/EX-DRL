'''Vanilla Option'''

from typing import Sequence, Optional

import numpy as np

from .base import (
    SDESimulator,
    ArrayTyped,
    Asset, TransactionCost, 
    PercentageTransactionCost
)
from patterns.lazyobj import LazyBase
from domain.pricer.bsm_pricers import vanilla_price_and_greeks
from config.config_loader import ConfigLoader

@ConfigLoader.register_class    
class VanillaOption(Asset, ArrayTyped, LazyBase):
    def __init__(self, 
                 sde: SDESimulator, 
                 transaction_cost: TransactionCost = PercentageTransactionCost(0.005),
                 call: Sequence[bool]=[],
                 moneyness: Sequence[float]=[],
                 ttm: Sequence[int]=[],
                 shares: Sequence[float]=[],
                 sim_moneyness_mean: float=1.,
                 sim_moneyness_std: float=0.2,
                 sim_ttms: Sequence[int]=[20,40,60],
                 sim_call: Sequence[bool]=[True]) -> None:
        """VanillaOption
        Vanilla Call or Put

        Args:
            sde (SDESimulator): Underlying SDE simulator
            transaction_cost (TransactionCost, optional): transaction cost function. Defaults to PercentageTransactionCost(0.005).
            call (list, optional): Vanilla option type is call. Defaults to [].
            moneyness (list, optional): option moneyness (K/S). Defaults to [].
            ttm (list, optional): option time to maturity. Defaults to [].
            shares (list, optional): number of shares. Defaults to [].
            sim_moneyness_mean (float, optional): mean of moneyness for simulation of new option for client portfolio. Defaults to 1.0.
            sim_moneyness_std (float, optional): std of moneyness for simulation of new option for client portfolio. Defaults to 0.2.
            sim_ttms (list, optional): time to maturity for simulation of new option for client portfolio. Defaults to [20,40,60].
            sim_call (list, optional): call or put for simulation of new option for client portfolio. Defaults to [True].
        """
        self.contract_size = 100.0
        self.init_params = {
            'call': np.array(call),
            'moneyness': np.array(moneyness),
            'ttm': np.array(ttm),
            'shares': np.array(shares)
        }
        self.sim_moneyness_mean = sim_moneyness_mean
        self.sim_moneyness_std = sim_moneyness_std
        self.sim_ttms = sim_ttms
        self.sim_call = sim_call
        
        Asset.__init__(self, sde=sde, transaction_cost=transaction_cost, shares=[])
        ArrayTyped.__init__(self, ['call', 'moneyness', 'ttm', 'shares', 'strike'])
        LazyBase.__init__(self)
        self.add(call=np.array(call), moneyness=np.array(moneyness), 
                 ttm=np.array(ttm), shares=np.array(shares))
        

    @LazyBase.lazy_func
    def price_and_greeks(self):
        """Black Scholes Formula

        Returns:
            np.ndarray: option price in the same shape of stock price S.
            np.ndarray: option delta in the same shape of stock price S.
            np.ndarray: option gamma in the same shape of stock price S.
            np.ndarray: option vega in the same shape of stock price S.
        """
        if len(self.ttm) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        return vanilla_price_and_greeks(self.ttm, self.sde.stock_price(), 
                                        self.sde.implied_vol(self.ttm, self.moneyness), 
                                        self.strike, self.call, self.sde.T)

    def aggregate_shares(self, call, moneyness, ttm, shares):
        # convert to structured array
        structured_array = np.array(list(zip(call, moneyness, ttm, shares)), 
                                    dtype=[('call', bool), ('moneyness', float), 
                                           ('ttm', int), ('shares', float)])
        # Find unique feature combinations and their indices
        unique_features, unique_indices = np.unique(structured_array[['call', 'moneyness', 'ttm']], 
                                                    return_inverse=True)

        # Initialize list for aggregated shares
        aggregated_shares = []

        # Loop over unique feature combinations
        for i in range(len(unique_features)):
            # Sum shares for this feature combination
            share_sum = np.sum(structured_array['shares'][unique_indices == i])
            aggregated_shares.append(share_sum)

        # Convert aggregated shares to numpy array
        aggregated_shares = np.array(aggregated_shares)

        # Only include options where the aggregated shares are not 0
        non_zero_indices = np.nonzero(aggregated_shares)[0]
        return dict(call=unique_features['call'][non_zero_indices], 
                    moneyness=unique_features['moneyness'][non_zero_indices], 
                    ttm=unique_features['ttm'][non_zero_indices], 
                    shares=aggregated_shares[non_zero_indices])
    

    def generate_options(self, num_options):
        option_calls = np.array(self.sim_call*num_options, dtype=bool)
        option_ttms = np.random.choice(self.sim_ttms, num_options)
        option_moneynss = np.random.normal(self.sim_moneyness_mean, self.sim_moneyness_std, num_options)
        option_shares = np.random.choice([1.0, -1.0], num_options)
        # aggregate back to back options' shares with same features, calls, ttms, moneyness
        return self.aggregate_shares(option_calls, option_moneynss, option_ttms, option_shares)

    def generate_atm_option(self):
        option_calls = np.array(self.sim_call, dtype=bool)
        option_ttms = np.array([self.sim_ttms[0]], dtype=int)
        option_moneynss = np.array([1.0], dtype=float)
        return VanillaOption(sde=self.sde, transaction_cost=self.transaction_cost,
                             call=option_calls, moneyness=option_moneynss, 
                             ttm=option_ttms, shares=[1.0])
        
    def get_value(self):
        return np.sum(self.price_and_greeks()[0] * self.shares)
    
    def get_delta(self):
        return np.sum(self.price_and_greeks()[1] * self.shares)
    
    def get_gamma(self):
        return np.sum(self.price_and_greeks()[2] * self.shares)
    
    def get_vega(self):
        return np.sum(self.price_and_greeks()[3] * self.shares)
    
    def trade_cost(self, option_ids=None):
        cost = 0.0
        if option_ids is None:
            option_ids = self.option_ids
        
        prices, _, _, _ = self.price_and_greeks()
        for option_id in option_ids:
            idx = self.get_idx(option_id)
            # trade option_id cost
            cost += (self.transaction_cost(share_price=prices[idx])*abs(self.shares[idx]))
        return cost
    
    def as_dict(self):
        return {
            'call': self.call,
            'moneyness': self.moneyness,
            'ttm': self.ttm,
            'shares': self.shares
        }

    def add(self, 
            call: Sequence[bool],
            moneyness: Sequence[float],
            ttm: Sequence[int],
            shares: Sequence[float],
            strike: Optional[Sequence[float]] = None) -> None:
        """
        add a single option

        Args:
            call (Sequence[bool]): Vanilla option type is call.
            moneyness (Sequence[float]): option moneyness (K/S) as of current time at SDE simulator step.
            ttm (Sequence[int]): option time to maturity.
            shares (Sequence[float]): number of shares.
            strike (Sequence[float], optional): option strike, can be derived from moneyness. Defaults to None.

        """    
        # filter ttm <= 0
        call = call[ttm >= 0]
        moneyness = moneyness[ttm >= 0]
        ttm = ttm[ttm >= 0]
        shares = shares[ttm >= 0]
        strike = strike[ttm >= 0]  if strike is not None else (self.sde.stock_price() * moneyness)
        shares = shares * self.contract_size
        option_ids = []
        for call_, moneyness_, ttm_, shares_, strike_ in zip(call, moneyness, ttm, shares, strike):
            option_ids.append(ArrayTyped.add(self, call=call_, moneyness=moneyness_, ttm=ttm_, shares=shares_, strike=strike_))
        self.clear()
        return self.trade_cost(option_ids=option_ids)

    def update(self, data: dict):
        """
        update option state after SDE simulation step
        move ttm and moneyness forward
        remove expired option
        
        Args:
            data (dict): Observable new state
        """
        self.ttm = self.ttm - 1
        s = self.sde.stock_price()
        self.moneyness = self.strike / s    # update moneyness as of current time
        # filter ttm <= 0 expired options
        self.remove(condition=self.ttm < 0)

    def exercise_dump(self):
        """
        Removes the expired portfolio positions to avoid position mismatch.
        It needs be called after the end of each step.
        """
        # to avoid the portfolio position mismatch after expired when update is called
        self.remove(condition=self.ttm == 0)
        
    def reset(self):
        for option_id in self.option_ids:
            self.remove(option_id)
        self.add(call=self.init_params['call'], moneyness=self.init_params['moneyness'], 
                 ttm=self.init_params['ttm'], shares=self.init_params['shares'],)
        