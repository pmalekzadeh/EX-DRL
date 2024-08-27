for vol in {1..8}; do
    python run_d3pg.py @run_configs/agents/d3pg.cfg --env_config=run_configs/envs/bsm_vanilla_eval.yaml --logger_prefix=policies/bsm_vanilla/bsm_vol0.3/robustness/bsm_vol0.$vol --eval_only --agent_path=policies/bsm_vanilla/bsm_vol0.3 --bsm_portfolio.client_trade_poisson_rate=1.0 --bsm_sde.vol=0.$vol --eval_env.vega_ratio=False
done
for vol in {1..8}; do
    python run_d3pg.py @run_configs/agents/d3pg.cfg --env_config=run_configs/envs/bsm_vanilla_eval.yaml --logger_prefix=policies/bsm_vanilla/bsm_vol0.6/robustness/bsm_vol0.$vol --eval_only --agent_path=policies/bsm_vanilla/bsm_vol0.6 --bsm_portfolio.client_trade_poisson_rate=1.0 --bsm_sde.vol=0.$vol --eval_env.vega_ratio=False
done
# for vol in {1..8}; do
#     python run_d3pg.py @run_configs/agents/d3pg.cfg --env_config=run_configs/envs/sabr_vanilla_eval.yaml --logger_prefix=policies/bsm_vanilla/dr_vol/robustness/sabr_vol0.$vol --eval_only --agent_path=policies/bsm_vanilla/dr_vol --sabr_portfolio.client_trade_poisson_rate=1.0 --sabr_sde.vol=0.$vol --eval_env.vega_ratio=False
# done
# for vol in {1..8}; do
#     python run_d3pg.py @run_configs/agents/d3pg.cfg --env_config=run_configs/envs/sabr_vanilla_eval.yaml --logger_prefix=policies/bsm_vanilla/dr_vol/robustness/sabr_volvol0.$vol --eval_only --agent_path=policies/bsm_vanilla/dr_vol --sabr_portfolio.client_trade_poisson_rate=1.0 --sabr_sde.volvol=0.$vol --eval_env.vega_ratio=False
# done
# for vol in {1..8}; do
#     python run_d3pg.py @run_configs/agents/d3pg.cfg --env_config=run_configs/envs/sabr_vanilla_eval.yaml --logger_prefix=policies/bsm_vanilla/dr_mixsde_vol_halfhalf/robustness/sabr_vol0.$vol --eval_only --agent_path=policies/bsm_vanilla/dr_mixsde_vol_halfhalf --sabr_portfolio.client_trade_poisson_rate=1.0 --sabr_sde.vol=0.$vol
# done
# for vol in {1..8}; do
#     python run_d3pg.py @run_configs/agents/d3pg.cfg --env_config=run_configs/envs/sabr_vanilla_eval.yaml --logger_prefix=policies/bsm_vanilla/dr_mixsde_vol_halfhalf/robustness/sabr_volvol0.$vol --eval_only --agent_path=policies/bsm_vanilla/dr_mixsde_vol_halfhalf --sabr_portfolio.client_trade_poisson_rate=1.0 --sabr_sde.volvol=0.$vol
# done
# for vol in {1..8}; do
#     python run_d3pg.py @run_configs/agents/d3pg.cfg --env_config=run_configs/envs/sabr_vanilla_fullstates_eval.yaml --logger_prefix=policies/bsm_vanilla/dr_mixsde_vol_sabrstates/robustness/sabr_vol0.$vol --eval_only --agent_path=policies/bsm_vanilla/dr_mixsde_vol_sabrstates --sabr_portfolio.client_trade_poisson_rate=1.0 --sabr_sde.vol=0.$vol
# done
# for vol in {1..8}; do
#     python run_d3pg.py @run_configs/agents/d3pg.cfg --env_config=run_configs/envs/sabr_vanilla_fullstates_eval.yaml --logger_prefix=policies/bsm_vanilla/dr_mixsde_vol_sabrstates/robustness/sabr_volvol0.$vol --eval_only --agent_path=policies/bsm_vanilla/dr_mixsde_vol_sabrstates --sabr_portfolio.client_trade_poisson_rate=1.0 --sabr_sde.volvol=0.$vol
# done

