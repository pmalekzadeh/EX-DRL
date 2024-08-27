'''Asset Interface'''

from abc import ABC, abstractmethod
from typing import Union, List
import uuid

import numpy as np

from domain.sde.base import SDESimulator
from patterns.observations import Observer
from config.config_loader import ConfigLoader


class TransactionCost(ABC):
    """
    TransactionCost(ABC)
    
    Class Interface for transaction cost function.

    """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

@ConfigLoader.register_class    
class PercentageTransactionCost(TransactionCost):
    def __init__(self, percent) -> None:
        self.percent = percent
        super().__init__()

    def __call__(self, share_price, *args, **kwargs):
        return self.percent * share_price

@ConfigLoader.register_class    
class FixedTransactionCost(TransactionCost):
    def __init__(self, cost) -> None:
        self.cost = cost
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.cost

class Asset(Observer, ABC):
    """
    Asset(ABC)

    Asset contains its price, and risk profiles.
    Any kind of asset can inherit this interface, i.e. stock, single option, portfolio constructed by multi-assets and etc.

    Methods
    -------
    2. get_value()
    3. get_delta()
    4. get_gamma()
    5. get_vega()
    """

    def __init__(self, sde: SDESimulator, 
                 transaction_cost: TransactionCost = FixedTransactionCost(0.0),
                 shares: Union[float, List[float]] = 0) -> None:
        self.sde = sde
        self.transaction_cost = transaction_cost
        self.shares = shares
        Observer.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def get_value(self) -> float:
        """asset value"""

    @abstractmethod
    def get_delta(self) -> float:
        """asset delta"""

    @abstractmethod
    def get_gamma(self) -> float:
        """asset gamma"""

    @abstractmethod
    def get_vega(self) -> float:
        """asset vega"""

    def trade_cost(self):
        return self.transaction_cost(share_price=self.get_value())*abs(self.shares)
        
    @abstractmethod
    def reset(self):
        pass

class ArrayTyped:
    def __init__(self, arr_keys):
        self.option_ids = np.array([])
        self.arr_keys = arr_keys
        for k in arr_keys:
            setattr(self, k, np.array([]))
        
    def add(self, **kwargs):
        option_ids = None
        
        for k in self.arr_keys:
            assert k in kwargs, f'Attribute {k} is mandatory and not found!'
            v = kwargs[k]
            setattr(self, k, np.append(getattr(self, k, np.array([])), v))
            if option_ids is None:
                # check if v is list liked value (list, or np.ndarray)
                # if not, convert it to list
                if not isinstance(v, list) or not isinstance(v, np.ndarray):
                    v = [v]
                option_ids = [uuid.uuid4() for _ in range(len(v))]
        self.option_ids = np.append(self.option_ids, option_ids)
        return option_ids
        
    def remove(self, option_id=None, condition=None):
        if option_id is not None:
            np_idx = self.get_idx(option_id)
            for arr_key in self.arr_keys:
                setattr(self, arr_key, np.delete(getattr(self, arr_key), np_idx))
            self.option_ids = np.delete(self.option_ids, np_idx)
        if condition is not None:
            for arr_key in self.arr_keys:
                setattr(self, arr_key, getattr(self, arr_key)[~condition])
            self.option_ids = self.option_ids[~condition]

    def __len__(self):
        return len(self.option_ids)

    def __getitem__(self, option_id):
        np_idx = self.get_idx(option_id)
        return {
            arr_key: getattr(self, arr_key)[np_idx] for arr_key in self.arr_keys
        }
    
    def get_idx(self, option_id):
        return np.where(option_id == self.option_ids)[0][0]
