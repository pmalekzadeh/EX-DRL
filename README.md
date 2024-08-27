# EX-DRL: Hedging Against Heavy Losses with EXtreme Distributional Reinforcement Learning

The EX-DRL algorithm, as detailed in the research paper ["EX-DRL: Hedging Against Heavy Losses with Extreme Distributional Reinforcement Learning"](https://arxiv.org/pdf/2408.12446), enhances Quantile Regression (QR)-based Distributional Reinforcement Learning (DRL) by improving extreme quantile predictions. It achieves this by modeling the tail of the loss distribution using a Generalized Pareto Distribution (GPD), which enhances the computation and reliability of risk metrics for developing hedging strategies in complex financial risk management.

This repository contains the code for EX-D4PG, which is developed by integrating our EX-DRL model with the Quantile Regression-based Distributed Distributional Deterministic Policy Gradients (QR-D4PG) proposed in ["Gamma and vega hedging using deep distributional reinforcement learning"](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1129370/full).

## Code Structure
```
EX-D4PG Codebase
│   run_d4pg.py - Run EX-D4PG model
└───agent
│   │   agent.py - EX-D4PG agent
│   │   distributional.py - distributional dependency for EX-D4PG
│   │   learning.py - learning module for EX-D4PG

└───env
│   │   trade_env.py - Trading Environment
│   │   test_trade_env.py - Test Trading Environment

└───run_configs
│   └───agents
    │   │    d4pg.cfg- EX-D4PG Configuration
```

## Paper Citation:
```
@article{malekzadeh2024ex,
  title={EX-DRL: Hedging Against Heavy Losses with EXtreme Distributional Reinforcement Learning},
  author={Malekzadeh, Parvin and Poulos, Zissis and Chen, Jacky and Wang, Zeyu and Plataniotis, Konstantinos N},
  journal={arXiv preprint arXiv:2408.12446},
  year={2024}
}

