from gym import spaces
import backtrader as bt
from btgym import BTgymDataset, BTgymBaseStrategy, BTgymEnv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pyts.image import GADF

MyCerebro = bt.Cerebro()
MyCerebro.addstrategy(
        BTgymBaseStrategy,
        #state_shape={'raw': spaces.Box(low=0, high=1, shape=(20,4))},
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
        render_enabled=False,
        verbose=1,
        )

max_episodes=10

def save_video(windows, video_name):
    print("saving sequence as ", video_name)
    image_size = 80
    gadf = GADF(image_size)
    X_gadf = gadf.fit_transform(np.array(windows))
    print(video_name+" : ")
    print(X_gadf.shape)
    fig = plt.figure()
    ims = []
    for i in range(len(X_gadf)):
        im = plt.imshow(X_gadf[i], cmap='rainbow', origin='lower', animated=True)
        ims.append([im])
    anim = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    anim.save(video_name)


for _ in range(0, max_episodes):
    done = False
    obs = env.reset()
    open_windows  = []
    high_windows  = []
    low_windows   = []
    close_windows = []
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action) 
        print("reward ", reward)
        raw = obs['raw']
        open_window  = []
        high_window  = []
        low_window   = []
        close_window = []
        for row in raw:
            open_window.append( row[0])
            high_window.append( row[1])
            low_window.append(  row[2])
            close_window.append(row[3])
        open_windows.append(open_window)
        high_windows.append(high_window)
        low_windows.append(low_window)
        close_windows.append(close_window)
        gadf = GADF(84)
        close_gadf = gadf.fit_transform(np.array([close_window]))
        state = close_gadf[0]

env.close()





