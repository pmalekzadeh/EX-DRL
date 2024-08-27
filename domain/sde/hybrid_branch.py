import numpy as np

from domain.sde.base import SDESimulator
from config.config_loader import ConfigLoader

@ConfigLoader.register_class
class HybridBranchSimulator(SDESimulator):
    def __init__(self,
                 base_simulator: SDESimulator,
                 branch_simulator: SDESimulator,
                 branch_num: int = 5,
                 branch_steps: int = 3,
                 **kwargs
                 ) -> None:
        """A hybrid SDE simulator
        The main trajectory is simulated by base_simulator.  
        At each step the branch_simulator is use to simulate a sub-branching trajectory with branch_steps ahead for branch_num times.
        The purpose of this class is to use the branching model-based simulator to complement the base simulator (e.g. empirical simulator) which has scarcy data. 

        Args:
            base_simulator (SDESimulator): The base simulator
            branch_simulator (SDESimulator): The branching simulator
            branch_num (int, optional): The number of branches at each step. Defaults to 5.
            branch_steps (int, optional): The number of steps for each branch. Defaults to 3.
        """
        assert base_simulator.T == branch_simulator.T, "Base and branch simulators must have the same T"
        assert base_simulator.dt == branch_simulator.dt, "Base and branch simulators must have the same dt"
        super().__init__(**kwargs)
        self.base_simulator = base_simulator
        self.branch_simulator = branch_simulator
        self.branch_num = branch_num
        self.branch_steps = branch_steps
        self.reset()

    def has_branch(self):
        return True
    
    def step(self):
        return super().step()