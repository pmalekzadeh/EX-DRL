'''Exotic Options'''
from typing import Sequence, Optional

import numpy as np

from .base import (
    SDESimulator,
    ArrayTyped,
    Asset, TransactionCost, 
    PercentageTransactionCost
)
from patterns.lazyobj import LazyBase
from config.config_loader import ConfigLoader
from domain.pricer.bsm_pricers import (
    dip_barrier_option_analytical, 
    dip_barrier_delta_analytical, 
    dip_barrier_gamma_analytical, 
    dip_barrier_vega_analytical
)

@ConfigLoader.register_class    
class BarrierDIPOption(Asset, ArrayTyped, LazyBase):
    def __init__(self, 
                 sde: SDESimulator, 
                 transaction_cost: TransactionCost = PercentageTransactionCost(0.005),
                 call: Sequence[bool]=[],
                 moneyness: Sequence[float]=[],
                 ttm: Sequence[int]=[],
                 shares: Sequence[float]=[],
                 barrier: Sequence[float]=[],
                 sim_moneyness_mean: float=1.,
                 sim_moneyness_std: float=0.2,
                 sim_ttms: Sequence[int]=[20,40,60],
                 sim_call: Sequence[bool]=[True],
                 sim_barrier_uniform: Sequence[float]=[0.6, 0.8]) -> None:
        """BarrierDIPOption
        Barrier Down-and-In Put Option

        Args:
            sde (SDESimulator): Underlying SDE simulator
            transaction_cost (TransactionCost, optional): transaction cost function. Defaults to PercentageTransactionCost(0.005).
            call (list, optional): Vanilla option type is call. Defaults to [].
            moneyness (list, optional): option moneyness (K/S). Defaults to [].
            ttm (list, optional): option time to maturity. Defaults to [].
            barrier (list, optional): option barrier level. Defaults to [].
            shares (list, optional): number of shares. Defaults to [].
            sim_moneyness_mean (float, optional): mean of moneyness for simulation of new option for client portfolio. Defaults to 1.0.
            sim_moneyness_std (float, optional): std of moneyness for simulation of new option for client portfolio. Defaults to 0.2.
            sim_ttms (list, optional): time to maturity for simulation of new option for client portfolio. Defaults to [20,40,60].
            sim_call (list, optional): call or put for simulation of new option for client portfolio. Defaults to [True].
            sim_barrier_uniform (list, optional): uniform distribution of barrier ratio for simulation of new option for client portfolio. Defaults to [0.6, 0.8].
        """
        self.contract_size = 100.0
        self.init_params = {
            'call': np.array(call),
            'moneyness': np.array(moneyness),
            'ttm': np.array(ttm),
            'barrier': np.array(barrier),
            'shares': np.array(shares)
        }
        self.sim_moneyness_mean = sim_moneyness_mean
        self.sim_moneyness_std = sim_moneyness_std
        self.sim_ttms = sim_ttms
        self.sim_call = sim_call
        self.sim_barrier_uniform = sim_barrier_uniform
        
        Asset.__init__(self, sde=sde, transaction_cost=transaction_cost, shares=[])
        ArrayTyped.__init__(self, 
                            ["call", "moneyness", "ttm",
                             "barrier", "shares", "strike", 
                             "barrier_crossing_indicator"])
        LazyBase.__init__(self)
        self.add(call=np.array(call), moneyness=np.array(moneyness),
                 ttm=np.array(ttm), shares=np.array(shares),
                 barrier=np.array(barrier))

    @LazyBase.lazy_func
    def price_and_greeks(self):
        """Black Scholes Formula

        Returns:
            np.ndarray: option price in the same shape of stock price S.
            np.ndarray: option delta in the same shape of stock price S.
            np.ndarray: option gamma in the same shape of stock price S.
            np.ndarray: option vega in the same shape of stock price S.
        """
        ttm = self.ttm
        T = self.sde.T
        # index
        # active option
        active_option = (ttm > 0).astype(np.uint0)
        
        args = (self.sde.stock_price(), self.strike, self.barrier,
                np.maximum(ttm, 1)/T, 0.0, 0.0, 
                self.sde.implied_vol(ttm, self.moneyness), 
                self.barrier_crossing_indicator)
        # down and in put
        active_bs_price = dip_barrier_option_analytical(*args)
        active_bs_delta = dip_barrier_delta_analytical(*args)
        active_bs_gamma = dip_barrier_gamma_analytical(*args)
        active_bs_vega = dip_barrier_vega_analytical(*args)

        # consolidate
        price = active_bs_price
        delta = active_option*active_bs_delta
        gamma = active_option*active_bs_gamma
        vega = active_option*active_bs_vega
        return price, delta, gamma, vega

    def aggregate_shares(self, calls, ttms, moneyness, barriers, shares):
        # Convert to structured array
        structured_array = np.array(list(zip(calls, ttms, moneyness, barriers, shares)),
                                    dtype=[('call', 'bool'), ('ttm', 'int'), ('moneyness', 'float'), 
                                        ('barrier', 'float'), ('shares', 'float')])

        # Find unique feature combinations and their indices
        unique_features, unique_indices = np.unique(structured_array[['call', 'ttm', 'moneyness', 'barrier']], 
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
                    ttm=unique_features['ttm'][non_zero_indices], 
                    moneyness=unique_features['moneyness'][non_zero_indices], 
                    barrier=unique_features['barrier'][non_zero_indices], 
                    shares=aggregated_shares[non_zero_indices])
    
    def generate_options(self, num_options):
        option_calls = np.array(self.sim_call*num_options, dtype=bool)
        option_ttms = np.random.choice(self.sim_ttms, num_options)
        option_moneynss = np.random.normal(self.sim_moneyness_mean, self.sim_moneyness_std, num_options)
        option_shares = np.random.choice([1.0, -1.0], num_options)
        option_barriers = np.random.uniform(self.sim_barrier_uniform[0], self.sim_barrier_uniform[1], num_options)*self.sde.stock_price()
        # aggregate back to back options' shares with same features, calls, ttms, moneyness, and barriers
        return self.aggregate_shares(option_calls, option_ttms, option_moneynss, option_barriers, option_shares)

    def as_dict(self):
        return dict(call=self.call, moneyness=self.moneyness,
                    ttm=self.ttm, shares=self.shares,
                    barrier=self.barrier)

    def get_value(self):
        return np.sum(self.price_and_greeks()[0] * self.shares)
    
    def get_delta(self):
        return np.sum(self.price_and_greeks()[1] * self.shares)
    
    def get_gamma(self):
        return np.sum(self.price_and_greeks()[2] * self.shares)
    
    def get_vega(self):
        return np.sum(self.price_and_greeks()[3] * self.shares)
    
    def get_barrier_crossing_indicator(self):
        return np.sum(self.barrier_crossing_indicator)/len(self.barrier_crossing_indicator)

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
    
    def add(self, 
            call: Sequence[bool],
            moneyness: Sequence[float],
            ttm: Sequence[int],
            barrier: Sequence[float],
            shares: Sequence[float],
            strike: Optional[Sequence[float]] = None) -> None:
        """
        add a sequence of options

        Args:
            call (Sequence[bool]): Vanilla option type is call.
            moneyness (Sequence[float]): option moneyness (K/S) as of current time at SDE simulator step.
            ttm (Sequence[int]): option time to maturity.
            barrier (Sequence[float]): option barrier level.
            shares (Sequence[float]): number of shares.
            strike (Sequence[float], optional): option strike, can be derived from moneyness. Defaults to None.

        """    
        # filter ttm <= 0
        call = call[ttm >= 0]
        moneyness = moneyness[ttm >= 0]
        ttm = ttm[ttm >= 0]
        barrier = barrier[ttm >= 0]
        shares = shares[ttm >= 0]
        s = self.sde.stock_price()
        strike = strike[ttm >= 0] if strike is not None else (s * moneyness)
        barrier_crossing_indicator = (s <= barrier).astype(np.uint0)
        ttm = ttm[ttm >= 0]
        shares = shares * self.contract_size
        option_ids = []
        for call_, moneyness_, ttm_, barrier_, shares_, strike_, barrier_crossing_indicator_ \
            in zip(call, moneyness, ttm, barrier, shares, strike, barrier_crossing_indicator):
            option_ids.append(ArrayTyped.add(self, call=call_, moneyness=moneyness_, ttm=ttm_, 
                                            barrier=barrier_, shares=shares_, strike=strike_, 
                                            barrier_crossing_indicator=barrier_crossing_indicator_))
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
        self.barrier_crossing_indicator = (self.barrier_crossing_indicator.astype(bool) | (s <= self.barrier)).astype(np.uint0)
        self.moneyness = self.strike / s
        # filter ttm <= 0 expired options
        self.remove(condition=self.ttm < 0)
    
    def exercise_dump(self):
        """
        Removes the expired portfolio positions to avoid position mismatch.
        It needs be called after the end of each step.
        """
        self.remove(condition=self.ttm == 0)    
    
    def reset(self):
        for option_id in self.option_ids:
            self.remove(option_id)
        self.add(call=self.init_params['call'], moneyness=self.init_params['moneyness'], 
                 ttm=self.init_params['ttm'], shares=self.init_params['shares'], 
                 barrier=self.init_params['barrier'])