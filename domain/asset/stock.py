'''Stock'''

from .base import (
    SDESimulator,
    Asset, TransactionCost, 
    PercentageTransactionCost
)
from patterns.lazyobj import LazyBase
from config.config_loader import ConfigLoader


@ConfigLoader.register_class
class Stock(Asset, LazyBase):
    def __init__(self, sde: SDESimulator, 
                 transaction_cost: TransactionCost = PercentageTransactionCost(0.0000),
                 shares: float = 0.0) -> None:
        self.init_shares = shares
        Asset.__init__(self, sde, transaction_cost=transaction_cost, shares=shares)
        LazyBase.__init__(self)
        self._s = sde.stock_price()

    @LazyBase.lazy_func
    def get_value(self):
        return self._s * self.shares
    
    @LazyBase.lazy_func
    def get_delta(self):
        return self.shares
    
    @LazyBase.lazy_func
    def get_gamma(self):
        return 0
    
    @LazyBase.lazy_func
    def get_vega(self):
        return 0
    
    def update(self, data: dict):
        self._s = data['s']

    def trade_cost(self, trade_shares=0.0):
        return self.transaction_cost(share_price=self._s)*abs(trade_shares)

    def reset(self):
        self.shares = self.init_shares
        self._s = self.sde.stock_price()
