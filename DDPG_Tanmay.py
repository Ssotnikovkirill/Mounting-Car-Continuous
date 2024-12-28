import gymnasium as gym  

import random
import numpy as np
import pickle
from tensorflow import keras
from keras.optimizers import Adam
import keras.backend as K

import os
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw

from Noise import OUNoise

import matplotlib.pyplot as plt

from collections import deque

import tensorflow as tf

live_plot = False

seed = 16
num_episodes = 2001
max_steps = 7000
min_steps = max_steps
exploring_starts = 1
average_of = 100

step_decay = 1
augment = 0

render_list = [0, 10, 25, 50, 100, 120, 150, 200, 350, 500, 700, 800]
save = True

class Actor(keras.Model):
    def __init__(self, num_states):
        super(Actor, self).__init__()
        self.fc1 = keras.layers.Dense(256, activation='relu')
        self.fc2 = keras.layers.Dense(256, activation='relu')
        self.fc3 = keras.layers.Dense(1, activation='tanh')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.fc3(x)

class Critic(keras.Model):
    def __init__(self, num_states, num_actions):
        super(Critic, self).__init__()
        self.fc1 = keras.layers.Dense(16, activation='relu')
        self.fc2 = keras.layers.Dense(32, activation='relu')
        self.action_value = keras.layers.Dense(32, activation='relu')
        self.combine = keras.layers.Concatenate()
        self.fc3 = keras.layers.Dense(256, activation='relu')
        self.fc4 = keras.layers.Dense(256, activation='relu')
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs):
        state, action = inputs
        state_out = self.fc2(self.fc1(state))
        action_out = self.action_value(action)
        combined = self.combine([state_out, action_out])
        x = self.fc3(combined)
        x = self.fc4(x)
        return self.output_layer(x)

class Agent:
    epsilon = 0
    epsilon_min = 0
    decay = 0.9

    learn_start = 1000
    gamma = 1
    alpha = 0.001
    tau = 0.005

    mem_len = 1e7
    memory = deque(maxlen=int(mem_len))

    def __init__(self, env, seed):
        self.env = env
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.actor = self.create_actor_model()
        self.target_actor = self.create_actor_model()
        self.noise = OUNoise(self.env.action_space.shape[0], seed, theta=0.2, sigma=0.5)
        self.critic = self.create_critic_model()
        self.target_critic = self.create_critic_model()
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())
        self.reset()

    def create_actor_model(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        inputs = keras.layers.Input(shape=(self.env.observation_space.shape[0],))  
        model = Actor(self.env.observation_space.shape[0])
        output = model(inputs)
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.alpha / 2,
            decay_steps=1e9,
            decay_rate=1)
        model = keras.Model(inputs, output)
        model.compile(loss="huber_loss", optimizer=Adam(learning_rate=lr_schedule))
        return model

    def create_critic_model(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        state_input = keras.layers.Input(shape=(self.env.observation_space.shape[0],))
        action_input = keras.layers.Input(shape=self.env.action_space.shape)
        model = Critic(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        output = model([state_input, action_input])
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.alpha,
            decay_steps=1e9,
            decay_rate=1)
        model = keras.Model([state_input, action_input], output)
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=lr_schedule))
        return model

    def replayBuffer(self, state, action, reward, next_state, terminal):
        self.memory.append([state, action, reward, next_state, terminal])

    @tf.function
    def replay(self, states, actions, rewards, next_states):
        # Приводим rewards к float32
        rewards = tf.cast(rewards, tf.float32)
        
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states, training=True)
            q_target = rewards + self.gamma * self.target_critic([next_states, target_actions], training=True)
            q_current = self.critic([states, actions], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(q_target - q_current))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions_pred = self.actor(states, training=True)
            q_current = self.critic([states, actions_pred], training=True)
            actor_loss = -tf.math.reduce_mean(q_current)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

    def update_weight(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def trainTarget(self):
        for target, source in zip(self.target_actor.variables, self.actor.variables):
            target.assign(source * self.tau + target * (1 - self.tau))
        for target, source in zip(self.target_critic.variables, self.critic.variables):
            target.assign(source * self.tau + target * (1 - self.tau))

    def sample2batch(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, terminals = zip(*samples)
        return tf.convert_to_tensor(states), tf.convert_to_tensor(actions), tf.convert_to_tensor(rewards), tf.convert_to_tensor(next_states)

    def train(self, state, action, reward, next_state, terminal, steps):
        self.replayBuffer(state, action, reward, next_state, terminal)
        if steps % 1 == 0 and len(self.memory) > self.learn_start:
            samples = self.sample2batch()
            if samples:
                states, actions, rewards, next_states = samples
                self.replay(states, actions, rewards, next_states)
        if steps % 1 == 0:
            self.trainTarget()

    def reset(self):
        self.epsilon *= self.decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def chooseAction(self, state, scale=False):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        if np.random.random() < self.epsilon:
            return np.random.uniform(-1, 1, size=self.env.action_space.shape)
        action = self.actor(state)
        noise = self.noise.sample()
        if scale:
            return np.clip(0.33 * tf.squeeze(action).numpy() + noise, -1, 1)
        return np.clip(tf.squeeze(action).numpy() + noise, -1, 1)

class DataStore:
    def __init__(self, averages, rewards):
        self.averages = averages
        self.rewards = rewards

def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20, im.size[1]/18), f'Episode: {episode_num}', fill=text_color)
    return im

def main(max_steps):
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
    agent = Agent(env, seed)
    rewards = np.zeros(num_episodes)
    rewards_av = deque(maxlen=int(average_of))
    averages = np.ones(num_episodes)
    print(seed)
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        frames = []
        
        for step in range(max_steps):
            if (episode in render_list) and save:
                frame = env.render()
                frames.append(_label_with_episode_number(frame, episode_num=episode))
            elif episode in render_list:
                env.render()
            if episode < exploring_starts:
                action = agent.chooseAction(state, True)
            else:
                action = agent.chooseAction(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            terminal = terminated or truncated
            # state is now a numpy array, so we don't need to reshape it
            total_reward += reward
            # Adjusting the reward augmentation for the new state format
            reward += ((state[0] + 1.2)**2) * augment
            agent.train(state, action, reward, next_state, terminal, step)
            state = next_state
            if terminal:
                break


        agent.noise.reset()

        rewards[episode:] = total_reward
        rewards_av.append(total_reward)
        averages[episode:] = np.mean(rewards_av)

        if np.mean(rewards_av) <= 90:  # step >= 199:
            print(
                "Failed to complete in episode {:4} with reward of {:8.3f} in {:5} steps, average reward of last {:4} episodes "
                "is {:8.3f}".format(episode, total_reward, step + 1, average_of, np.mean(rewards_av)))

        else:
            print("Completed in {:4} episodes, with reward of {:8.3f}, average reward of {:8.3f}".format(episode, total_reward, np.mean(rewards_av)))
            #break

        if live_plot:
            plt.subplot(2, 1, 1)
            plt.plot(averages)
            plt.subplot(2, 1, 2)
            plt.plot(rewards)
            plt.pause(0.0001)
            plt.clf()

        max_steps = int(max(min_steps, max_steps*step_decay))

        if episode % 25 == 0:
            data = DataStore(averages, rewards)
            with open('data_ddpg.pk1', 'wb') as handle:
                pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

        env.close()
        if frames:
            imageio.mimwrite(os.path.join('./videos/', 'agent_ep_{}.gif'.format(episode)), frames, fps=30)
            del frames

    with open('data_ddpg.pk1', 'wb') as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(max_steps)
