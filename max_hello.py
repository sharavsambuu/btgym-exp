from gym import spaces
import backtrader as bt
from btgym import BTgymDataset, BTgymBaseStrategy, BTgymEnv

MyCerebro = bt.Cerebro()
MyCerebro.addstrategy(
        BTgymBaseStrategy,
        state_shape={'raw': spaces.Box(low=0, high=1, shape=(20,4))},
        skip_frame=5,
        state_low=None,
        state_high=None,
        drawdown_call=50,
        )

MyCerebro.broker.setcash(100.0)
MyCerebro.broker.setcommission(commission=0.001)
MyCerebro.addsizer(bt.sizers.SizerFix, stake=10)
MyCerebro.addanalyzer(bt.analyzers.DrawDown)

MyDataset = BTgymDataset(
        filename="./btgym/examples/data/DAT_ASCII_EURUSD_M1_2016.csv",
        start_weekdays=[0,1,2,4],
        start_00=True,
        episode_duration={'days':0, 'hours':23, 'minutes': 55},
        time_gap={'hours': 5},
        )

MyEnvironment = BTgymEnv(
        dataset=MyDataset,
        engine=MyCerebro,
        port=5555,
        verbose=1,
        )
