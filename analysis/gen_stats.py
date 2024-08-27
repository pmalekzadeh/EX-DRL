import os
import pandas as pd
from pathlib import Path

from absl import flags

FLAGS = flags.FLAGS


def generate_stat(eval_env_log, eval_quantiles, name_prefix=''):
    fpath = Path(eval_env_log)
    log_folder = str(fpath.parent.absolute()) + '/../../summary'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    eval_df = pd.read_csv(eval_env_log)
    eval_loop_df = eval_df[['episode', 'step_pnl_3']
                           ].groupby('episode').aggregate('sum')
    eval_loop_df.columns = ['episode_return']
    res = {}
    res['mean'], res['std'] = [eval_loop_df['episode_return'].mean()], [
        eval_loop_df['episode_return'].std()]
    try:
        res['mean-std'] = [res['mean'][0] - FLAGS.std_coef * res['std'][0]]
    except:
        res['mean-std'] = [res['mean'][0] - 1.645 * res['std'][0]]
    for q in eval_quantiles:
        res[f'var{q*100:.0f}'] = [eval_loop_df['episode_return'].quantile(1-q)]
        res[f'cvar{q*100:.0f}'] = [eval_loop_df.loc[eval_loop_df['episode_return']
                                                    <= eval_loop_df['episode_return'].quantile(1-q), 'episode_return'].mean()]

    res_df = pd.DataFrame(res)
    res_df.to_csv(f'{log_folder}/{name_prefix}_stats.csv', index=False)
    return res_df
