from gym import spaces
import backtrader as bt
from btgym import BTgymDataset, BTgymBaseStrategy, BTgymEnv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pyts.image import GADF
import math
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
from skimage import transform

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
        render_enabled=False,
        verbose=0,
        )

# Environment Parameters
n_actions = 4
n_epochs = 5000
n = 0
average = []
step = 1
batch_size = 5000
render = True

# Hyper Parameters
alpha = 1e-4
gamma = 0.99
normalize_r = True
save_path='models/pg-trader.ckpt'
value_scale = 0.5
entropy_scale = 0.00
gradient_clip = 40

# Apply discount to episode rewards & normalize
def discount(r, gamma, normal):
    discount = np.zeros_like(r)
    G = 0.0
    for i in reversed(range(0, len(r))):
        G = G * gamma + r[i]
        discount[i] = G
    # Normalize 
    if normal:
        mean = np.mean(discount)
        std = np.std(discount)
        discount = (discount - mean) / (std)
    return discount

# Conv Layers
convs = [16,32]
kerns = [8,8]
strides = [4,4]
pads = 'valid'
fc = 256
activ = tf.nn.elu

# Function for resizing image
def resize(image):
    # Greyscale Image
    x = np.mean(image,-1)
    # Normalize Pixel Values
    x = x/255
    x = transform.resize(x, [84,84])
    return(x)

# Tensorflow Variables
X = tf.placeholder(tf.float32, (None,84,84,1), name='X')
Y = tf.placeholder(tf.int32, (None,), name='actions')
R = tf.placeholder(tf.float32, (None,), name='reward')
N = tf.placeholder(tf.float32, (None), name='episodes')
D_R = tf.placeholder(tf.float32, (None,), name='discounted_reward')

# Policy Network
conv1 = tf.layers.conv2d(
        inputs = X,
        filters = convs[0],
        kernel_size = kerns[0],
        strides = strides[0],
        padding = pads,
        activation = activ,
        name='conv1')

conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters = convs[1],
        kernel_size = kerns[1],
        strides = strides[1],
        padding = pads,
        activation = activ,
        name='conv2')

flat = tf.layers.flatten(conv2)

dense = tf.layers.dense(
        inputs = flat, 
        units = fc, 
        activation = activ,
        name = 'fc')

logits = tf.layers.dense(
         inputs = dense, 
         units = n_actions, 
         name='logits')

value = tf.layers.dense(
        inputs=dense, 
        units = 1, 
        name='value')

calc_action = tf.multinomial(logits, 1)
aprob = tf.nn.softmax(logits)
action_logprob = tf.nn.log_softmax(logits)

tf.trainable_variables()


def rollout(batch_size, render):
    
    states, actions, rewards, rewardsFeed, discountedRewards = [], [], [], [], []
    env.reset()
    state = np.random.rand(84, 84)
    episode_num = 0 
    #action_repeat = 3
    action_repeat = 1
    reward = 0
    
    while True: 
        
        #if render:
            #env.render()
        
        # Run State Through Policy & Calculate Action
        feed = {X: state.reshape(1, 84, 84, 1)}
        action = sess.run(calc_action, feed_dict=feed)
        action = int(action[0][0])
        
        # Perform Action
        for i in range(action_repeat):
            state2, reward2, done, info = env.step(action)
            if math.isnan(reward2):
                reward2 = 0.0
            reward += reward2
            if done:
                break
        
        # Store Results
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        
        # Update Current State
        reward = 0
        raw = state2['raw']
        close_window = []
        for row in raw:
            close_window.append(row[3])
        gadf = GADF(84)
        close_gadf = gadf.fit_transform(np.array([close_window]))
        state = close_gadf[0]
        
        if done:
            # Track Discounted Rewards
            rewardsFeed.append(rewards)
            discountedRewards.append(discount(rewards, gamma, normalize_r))
            
            if len(np.concatenate(rewardsFeed)) > batch_size:
                break
                
            # Reset Environment
            rewards = []
            env.reset()
            state = np.random.rand(84, 84)
            episode_num += 1
                         
    return np.stack(states), np.stack(actions), np.concatenate(rewardsFeed), np.concatenate(discountedRewards), episode_num


mean_reward = tf.divide(tf.reduce_sum(R), N)

# Define Losses
pg_loss = tf.reduce_mean((D_R - value) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
value_loss = value_scale * tf.reduce_mean(tf.square(D_R - value))
entropy_loss = -entropy_scale * tf.reduce_sum(aprob * tf.exp(aprob))
loss = pg_loss + value_loss - entropy_loss

# Create Optimizer
optimizer = tf.train.AdamOptimizer(alpha)
grads = tf.gradients(loss, tf.trainable_variables())
grads, _ = tf.clip_by_global_norm(grads, gradient_clip) # gradient clipping
grads_and_vars = list(zip(grads, tf.trainable_variables()))
train_op = optimizer.apply_gradients(grads_and_vars)

# Initialize Session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tmp/dpg")
tf.summary.scalar('Total_Loss', loss)
tf.summary.scalar('PG_Loss', pg_loss)
tf.summary.scalar('Entropy_Loss', entropy_loss)
tf.summary.scalar('Value_Loss', value_loss)
tf.summary.scalar('Reward_Mean', mean_reward)
tf.summary.histogram('Conv1', tf.trainable_variables()[0])
tf.summary.histogram('Conv2', tf.trainable_variables()[2])
tf.summary.histogram('FC', tf.trainable_variables()[4])
tf.summary.histogram('Logits', tf.trainable_variables()[6])
tf.summary.histogram('Value', tf.trainable_variables()[8])
write_op = tf.summary.merge_all()

# Load model if exists
saver = tf.train.Saver(tf.global_variables())
load_was_success = True 
try:
    save_dir = '/'.join(save_path.split('/')[:-1])
    ckpt = tf.train.get_checkpoint_state(save_dir)
    load_path = ckpt.model_checkpoint_path
    saver.restore(sess, load_path)
except:
    print("No saved model to load. Starting new session")
    writer.add_graph(sess.graph)
    load_was_success = False
else:
    print("Loaded Model: {}".format(load_path))
    saver = tf.train.Saver(tf.global_variables())
    step = int(load_path.split('-')[-1])+1

while step < n_epochs+1:
    # Gather Training Data
    print('Epoch', step)
    s, a, r, d_r, n = rollout(batch_size,render)
    mean_reward = np.sum(r)/n
    average.append(mean_reward)
    print('Training Episodes: {}  Average Reward: {:4.2f}  Total Average: {:4.2f}'.format(n, mean_reward, np.mean(average)))
          
    # Update Network
    print("UPDATING NEURAL NETWORK >>>>>")
    sess.run(train_op, feed_dict={X:s.reshape(len(s),84,84,1), Y:a, D_R: d_r})
          
    # Write TF Summaries
    summary = sess.run(write_op, feed_dict={X:s.reshape(len(s),84,84,1), Y:a, D_R: d_r, R: r, N:n})
    writer.add_summary(summary, step)
    writer.flush()
          
    # Save Model
    if step % 10 == 0:
          print("SAVED MODEL")
          saver.save(sess, save_path, global_step=step)
          
    step += 1

#state = resize(env.reset())
#prob, val = sess.run([aprob, value], feed_dict={X: state.reshape(1, 84, 84, 1)})

#print('Turn Right: {:4.2f}  Turn Left: {:4.2f}'.format(prob[0][0],prob[0][1]))
#print('Approximated State Value: {:4.4f}'.format(val[0][0]))




