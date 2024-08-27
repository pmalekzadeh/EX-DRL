
"""D4PG agent implementation."""

from acme import core
from acme import types

import dm_env
import numpy as np

# # TODO Update new env
# class VegaHedgeAgent(core.Actor):
#     def __init__(self, running_env) -> None:
#         self.env = running_env
#         super().__init__()

#     def select_action(self, observation: types.NestedArray) -> types.NestedArray:
#         episode = self.env.sim_episode
#         t = self.env.t
#         current_vega = observation[3]
#         hedge_option = self.env.portfolio.hed_port.options[episode, t]
#         hed_share = -current_vega / \
#             hedge_option.vega_path[t]/self.env.portfolio.utils.contract_size
#         # action constraints
#         gamma_action_bound = -self.env.portfolio.get_gamma(
#             t)/self.env.portfolio.hed_port.options[episode, t].gamma_path[t]/self.env.portfolio.utils.contract_size
#         action_low = [0, gamma_action_bound]
#         action_high = [0, gamma_action_bound]
#         if self.env.vega_state:
#             # vega bounds
#             vega_action_bound = -self.env.portfolio.get_vega(
#                 t)/self.env.portfolio.hed_port.options[episode, t].vega_path[t]/self.env.portfolio.utils.contract_size
#             action_low.append(vega_action_bound)
#             action_high.append(vega_action_bound)
#         low_val = np.min(action_low)
#         high_val = np.max(action_high)
#         alpha = (hed_share - low_val)/(high_val - low_val)
#         return np.array([alpha])

#     def observe_first(self, timestep: dm_env.TimeStep):
#         pass

#     def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
#         pass

#     def update(self, wait: bool = False):
#         pass


class VegaHedgeAgent(core.Actor):
    def __init__(self, env, hedge_ratio=1.0) -> None:
        self.env = env
        assert 'port_vega' in self.env.env_states, 'vega state is not enabled'
        self.hedge_ratio = hedge_ratio
        super().__init__()

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        hedging_option = self.env.portfolio.hedging_options.generate_atm_option()
        hedging_vega_shares = -self.env.portfolio.get_vega()/hedging_option.get_vega() * self.hedge_ratio
        hedging_ratio = self.env.inverse_transform_action(
            np.array([hedging_vega_shares])
        )
        return np.array([hedging_ratio])

    def observe_first(self, timestep: dm_env.TimeStep):
        pass

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        pass

    def update(self, wait: bool = False):
        pass

class GammaHedgeAgent(core.Actor):
    def __init__(self, env, hedge_ratio=1.0) -> None:
        self.env = env
        self.hedge_ratio = hedge_ratio
        super().__init__()

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        hedging_option = self.env.portfolio.hedging_options.generate_atm_option()
        hedging_gamma_shares = -self.env.portfolio.get_gamma()/hedging_option.get_gamma() * self.hedge_ratio
        hedging_ratio = self.env.inverse_transform_action(
            np.array([hedging_gamma_shares])
        )
        return np.array([hedging_ratio])

    def observe_first(self, timestep: dm_env.TimeStep):
        pass

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        pass

    def update(self, wait: bool = False):
        pass


class DeltaHedgeAgent(core.Actor):
    def __init__(self, env, hedge_ratio=1.0) -> None:
        self.env = env
        self.hedge_ratio = hedge_ratio
        super().__init__()

    def select_action(self, observation: types.NestedArray = None) -> types.NestedArray:
        hedging_ratio = self.env.inverse_transform_action(
            np.array([0.0])
        )
        return np.array([hedging_ratio])

    def observe_first(self, timestep: dm_env.TimeStep):
        pass

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        pass

    def update(self, wait: bool = False):
        pass


# class OneDayOptimalDeltaHedgeAgent(core.Actor):
#     def __init__(self, running_env, no_hedge=False, one_day_num_sim=5_000,
#                  obj_func='var', std_coef=1.645, threshold=0.99) -> None:
#         self.env = running_env
#         self.no_hedge = no_hedge
#         self.one_day_num_sim = one_day_num_sim
#         self.obj_func = obj_func
#         self.std_coef = std_coef
#         self.threshold = threshold
#         super().__init__()

#     def select_action(self, observation: types.NestedArray) -> types.NestedArray:
#         if self.no_hedge:
#             return np.array([0])
#         utils = self.env.utils
#         # GBM info
#         S = observation[0]*np.ones((self.one_day_num_sim, 1))
#         mu = utils.mu
#         vol = self.env.portfolio.get_vol(
#             self.env.t)*np.ones((self.one_day_num_sim, 1))
#         dt = utils.dt

#         # Option info
#         num_contract_to_add = utils.num_conts_to_add
#         contract_size = utils.contract_size
#         K = utils.K*np.ones((self.one_day_num_sim, 1))
#         H = utils.H
#         tau = (utils.init_ttm - self.env.t) / \
#             utils.T*np.ones((self.one_day_num_sim, 1))
#         r = utils.r
#         q = utils.q
#         sigma = vol
#         n_obs = utils.init_ttm - self.env.t

#         # Simulate 5000 times for next day stock price to compile distribtution
#         z = np.random.normal(size=(self.one_day_num_sim, 1))
#         a_prices = S * np.exp((mu - (vol**2) / 2)
#                               * dt + vol * np.sqrt(dt) * z)
#         # Barrier crossing indicator
#         barrier_crossing_indicator = observation[2] * \
#             np.ones((self.one_day_num_sim, 1))
#         barrier_crossing_indicator_new = np.where(
#             a_prices - H > 0, barrier_crossing_indicator, 1)
#         # Construct barrier option p&l distribution
#         grid_size = int((self.max_hedge_ratio -
#                         self.min_hedge_ratio + 3*max(4-n_obs, 0))/self.step_size)
#         deltas = np.linspace(self.min_hedge_ratio, self.max_hedge_ratio +
#                              3*max(4-n_obs, 0), num=grid_size+1).reshape(1, grid_size+1)
#         prev_action = self.env.portfolio.prev_action * \
#             np.ones((1, grid_size+1))
#         bsm_barrier_delta_shares = utils.barrier_delta_analytical(
#             S, K, H, tau, r, q, sigma, barrier_crossing_indicator, n_obs) * \
#             num_contract_to_add*(-1) * contract_size

#         liab_pnl = num_contract_to_add * contract_size * (utils.barrier_option_analytical(a_prices, K, H, tau-dt, r, q, sigma, barrier_crossing_indicator_new, n_obs-1)
#                                                           - utils.barrier_option_analytical(S, K, H, tau, r, q, sigma, barrier_crossing_indicator, n_obs))
#         hed_pnl = (a_prices - S) * bsm_barrier_delta_shares * deltas
#         transaction_cost = S * \
#             np.abs(bsm_barrier_delta_shares *
#                    deltas - prev_action) * utils.spread
#         total_pnl = (liab_pnl + hed_pnl - transaction_cost).T

#         obj = 0
#         if self.obj_func == 'meanstd':
#             obj = np.mean(total_pnl, axis=1) - self.std_coef * \
#                 np.std(total_pnl, axis=1)
#         if self.obj_func == 'var':
#             obj = np.quantile(total_pnl, q=1-self.threshold, axis=1)
#         if self.obj_func == 'cvar':
#             var = np.quantile(total_pnl, q=1-self.threshold,
#                               axis=1, keepdims=True)
#             obj = np.sum((total_pnl <= var).astype(int) * total_pnl,
#                          axis=1)/np.sum((total_pnl <= var).astype(int), axis=1)

#         hedge_ratio = deltas.squeeze()[np.argmax(obj)]
#         return np.array([hedge_ratio*bsm_barrier_delta_shares[0, 0]])

#     def observe_first(self, timestep: dm_env.TimeStep):
#         pass

#     def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
#         pass

#     def update(self, wait: bool = False):
#         pass
