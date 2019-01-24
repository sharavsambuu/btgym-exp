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

state = np.random.rand(84, 84)
feed = {X: state.reshape(1, 84, 84, 1)}
action = sess.run(calc_action, feed_dict=feed)
action = int(action[0][0])
print(action)

sess.close()