'''Portfolio
Object representing portfolio of assets and used in environment
'''

from typing import List, Optional, Union
import numpy as np

from .base import (
    SDESimulator,
    Asset
)
from patterns.lazyobj import LazyBase
from domain.asset.stock import Stock
from domain.asset.vanilla_option import VanillaOption
from domain.asset.barrier_option import BarrierDIPOption
from config.config_loader import ConfigLoader


@ConfigLoader.register_class
class Portfolio(Asset, LazyBase):
    def __init__(self, sde: SDESimulator,
                 client_options: Union[VanillaOption, BarrierDIPOption],
                 hedging_options: VanillaOption,
                 client_trade_poisson_rate: float = 1.0,
                 ) -> None:
        """Portfolio object
        The events sequences at each step are:  
        Event 1. new arrival client options are generated
        Event 2. hedging actions are taken (including stock trade and hedging option trade)
        Event 3. exercise expired options 
        ----------- Up-to-now the portfolio updates are done -----------
        Event 4. sde steps over to next time step to update market states 
        Event 5. pnl of portfolio is updated (keep portfolio position unchanged)
        ----------- Up-to-now the events within a step are done -----------
        In environment it loops for each step from Event 1 to 5. 
        Events are paddled by Environment.step() and Environment.reset().

        Args:
            sde (SDESimulator): _description_
            client_options (_type_): _description_
            hedging_options (_type_): _description_
            client_trade_poisson_rate (float, optional): _description_. Defaults to 1.0.
        """
        Asset.__init__(self, sde)
        LazyBase.__init__(self)
        sde.add_observer(self)
        self.stock: Stock = Stock(sde)
        self.client_options: Union[VanillaOption, BarrierDIPOption] = client_options
        self.hedging_options: Union[VanillaOption, BarrierDIPOption] = hedging_options
        self.poisson_rate = client_trade_poisson_rate

    def simulate_client_trade(self) -> float:
        """
        Function to simulate client trade.
        Event 1. new arrival client options are generated
        """
        # => Reset 1. client options lazy properties are reset
        self.client_options.clear()
        num_options = np.random.poisson(self.poisson_rate)
        if num_options == 0:
            return 0.0
        new_options = self.client_options.generate_options(num_options)
        self.client_options.add(**new_options)
        return new_options["shares"].sum()
        

    def trade_hedging(self, hedging_option_share: float = 0., stock_shares: Optional[float] = None) -> float:
        """
        Event 2. hedging actions are taken (including stock trade and hedging option trade)
        Event 3. exercise expired options
        """
        # Event 2. hedging actions are taken
        # hedging option trade
        # stock trade
        if hedging_option_share != 0.:
            hedging_option = self.hedging_options.generate_atm_option().as_dict()
            hedging_option["shares"] = np.array([hedging_option_share])
            # => Reset 2. hedging options lazy properties are reset
            self.hedging_options.clear()
            option_trade_cost = self.hedging_options.add(**hedging_option)
        else:
            option_trade_cost = 0.0
        
        self.stock.clear()
        if stock_shares is not None:
            stock_trade_cost = self.stock.trade_cost(stock_shares - self.stock_shares)
            self.stock_shares = stock_shares
        else:
            # auto delta
            stock_trade_cost = self.stock.trade_cost(self.get_delta())
            self.stock.shares = self.stock.shares - self.get_delta()
        # update stock position value after hedging
        self.stock.get_value()
        # Event 3. exercise expired options
        self.client_options.clear()
        self.client_options.exercise_dump()
        self.client_options.price_and_greeks()
        self.hedging_options.clear()
        self.hedging_options.exercise_dump()
        self.hedging_options.price_and_greeks()
        return option_trade_cost + stock_trade_cost

    def get_value(self):
        return self.stock.get_value() + self.client_options.get_value() + self.hedging_options.get_value()
    
    def get_delta(self):
        return self.stock.get_delta() + self.client_options.get_delta() + self.hedging_options.get_delta()
    
    def get_gamma(self):
        return self.stock.get_gamma() + self.client_options.get_gamma() + self.hedging_options.get_gamma()
    
    def get_vega(self):
        return self.stock.get_vega() + self.client_options.get_vega() + self.hedging_options.get_vega()
    
    def _reset_underlying_lazy(self):
        self.stock.clear()
        self.client_options.clear()
        self.hedging_options.clear()

    @LazyBase.lazy_func
    def get_pnl(self) -> float:
        """Event 5. pnl of portfolio is updated (keep portfolio position unchanged)

        Returns:
            float: Step pnl
        """
        # store the values from previous step
        # because underlying properties are lazy, the SDE step are not counted in the property values 
        prev_stock_value = self.stock.get_value()
        prev_client_options_value = self.client_options.get_value()
        prev_hedging_options_value = self.hedging_options.get_value()
        # reset underlying lazy properties to adopt SDE step of market states
        self._reset_underlying_lazy()
        self.stock_pnl = self.stock.get_value() - prev_stock_value
        self.client_options_pnl = self.client_options.get_value() - prev_client_options_value
        self.hedging_options_pnl = self.hedging_options.get_value() - prev_hedging_options_value
        return self.stock_pnl + self.client_options_pnl + self.hedging_options_pnl
    
    def update(self, data: dict):
        if data['action'] == 'step':
            self.step(data)
        elif data['action'] == 'reset':
            self.reset()
    
    def step(self, data: dict):
        """
        Event 4. SDE simulator step over a day
        emit signal to underlying assets to get the step updates
        it makes changes of underlying states such as moving ttm, moneyness, drop expired options and etc.

        Args:
            data (dict): data emitted from SDE simulator
        """
        self.stock.update(data)
        self.client_options.update(data)
        self.hedging_options.update(data)

    def reset(self):
        """Reset portfolio states to initial states
        Reset underlying states to initial states
        """
        self.stock.reset()
        self.client_options.reset()
        self.hedging_options.reset()