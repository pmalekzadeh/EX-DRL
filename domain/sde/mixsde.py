from typing import Any, Sequence

import numpy as np

from domain.sde.base import SDESimulator
from config.config_loader import ConfigLoader


@ConfigLoader.register_class
class MixSDESimulator(SDESimulator):
    def __init__(self, 
                 simulators: Sequence[SDESimulator], 
                 probabilities: Sequence[float], **kwargs):
        assert len(simulators) == len(probabilities), "Length of simulators and probabilities must be the same"
        assert sum(probabilities) == 1.0, "Sum of probabilities must be 1.0"
        for simulator in simulators:
            assert simulator.T == simulators[0].T
            assert simulator.dt == simulators[0].dt
        super().__init__(**kwargs)
        self.simulators = simulators
        self.probabilities = probabilities
        self.current_simulator = None
        self.reset()

    def step(self):
        return self.current_simulator.step()

    def stock_price(self):
        return self.current_simulator.stock_price()

    def implied_vol(self, ttm, moneyness):
        return self.current_simulator.implied_vol(ttm, moneyness)

    def reset(self, *args, **kwargs):
        self.current_simulator = np.random.choice(self.simulators, p=self.probabilities)
        return self.current_simulator.reset(*args, **kwargs)

    def add_observer(self, observer):
        for simulator in self.simulators:
            simulator.add_observer(observer)
    
    def __getattr__(self, name: str):
        # Expose any other attributes of the underlying environment.
        return getattr(self.current_simulator, name)
