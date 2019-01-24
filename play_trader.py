from gym import spaces
import backtrader as bt
from btgym import BTgymDataset, BTgymBaseStrategy, BTgymEnv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pyts.image import GADF

import tensorflow as tf
import numpy as np

saver = tf.train.import_meta_graph('./models/pg-trader.ckpt-1560.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "./models/pg-trader.ckpt-1560")
X = graph.get_tensor_by_name("X:0")
logits = graph.get_tensor_by_name('logits/kernel:0')
calc_action = tf.multinomial(logits, 1)

MyCerebro = bt.Cerebro()
MyCerebro.addstrategy(
        BTgymBaseStrategy,
        state_shape={'raw': spaces.Box(low=0, high=1, shape=(120,4))},
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
env = BTgymEnv(
        dataset=MyDataset,
        engine=MyCerebro,
        port=5555,
        #render_enabled=False,
        verbose=1,
        )

done = False
current_state = env.reset()

while not done:
    raw = current_state['raw']
    close_window = []
    for row in raw:
        close_window.append(row[3]) # close price
    gadf = GADF(84)
    close_gadf = gadf.fit_transform(np.array([close_window]))
    state=close_gadf[0]
    # make decision with neural network trained with RL
    feed = {X: state.reshape(1, 84, 84, 1)}
    action = sess.run(calc_action, feed_dict=feed)
    action = int(action[0][0])

    obs, reward, done, info = env.step(action)

    current_state=obs
    
    print('ACTION: {}\nREWARD: {}\nINFO: {}'.format(action, reward, info))

env.close()

