import numpy
import gym
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env.reset()

e = 0.05

inputs = tf.placeholder(shape = [4], dtype = tf.float32)
W = tf.Variable(tf.random_uniform([4, 2], 0, 0.01))
Qout = tf.matmul(tf.reshape(inputs, (1, 4)), W)
Qmax = tf.reduce_max(Qout)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape = [1], dtype = tf.float32)
loss = tf.nn.l2_loss(nextQ - Qmax) + 0.01 * tf.nn.l2_loss(W)
trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
updateModel = trainer.minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

times = []

for i_episode in range(2000):
    observation = env.reset()
    for t in range(200):
        #env.render()

        Q = sess.run(Qout, feed_dict = { inputs : observation }).flatten()
        action = numpy.argmax(Q)

        if numpy.random.rand() < e:
            action = env.action_space.sample()
            
        nobservation, reward, done, info = env.step(action)

        nQ = 0.95 * sess.run(Qmax, feed_dict = { inputs : nobservation }) + reward

        _, l = sess.run([updateModel, loss], feed_dict = { inputs : observation, nextQ : [nQ] })

        observation = nobservation

        #print l
        #print Q.max(), nQ
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            times.append(t + 1)
            print sess.run(W)
            break

plt.plot(times)
plt.show()

