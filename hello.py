import gym
from gym import spaces 
import backtrader as bt
from btgym import BTgymDataset, BTgymBaseStrategy, BTgymEnv

env_params =dict(
        filename='./btgym/examples/data/DAT_ASCII_EURUSD_M1_2016.csv',
        episode_duration={'days': 2, 'hours': 23, 'minutes': 55},
        drawdow_call=50,
        state_shape={'raw': spaces.Box(low=0, high=1, shape=(20, 4))},
        port=5555,
        verbose=1,
        )

gym.envs.register(id='backtrader-v5555', entry_point='btgym:BTgymEnv', kwargs=env_params,)
env = gym.make('backtrader-v5555')

#print(env.get_stat())
env.reset()
res = env.step(2)
print(res)
env.close()


