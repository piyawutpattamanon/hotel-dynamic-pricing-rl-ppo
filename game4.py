import numpy as np
import gym
from gym import spaces
import tensorflow as tf

REWARD_CAP = 0
STATE_SIZE = 3 + REWARD_CAP

class SimpleGameEnv(gym.Env):
    def __init__(self):
        super(SimpleGameEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: rock, 1: paper, 2: scissors
        self.observation_space = spaces.Discrete(3)  # Our last action
        self.state = 0
        self.episode_length = 10
        self.step_count = 0
        self.last_three_actions = []
        self.accumulated_reward = 0
        self.historical_rewards = []

    def reset(self):
        self.state = (np.zeros(STATE_SIZE), np.zeros(STATE_SIZE))
        self.step_count = 0
        self.last_three_actions = []
        self.accumulated_reward = 0
        return self.state

    def step(self, action1, action2):
        self.step_count += 1

        market_preference1 = np.random.normal(np.zeros(3), 0.3)
        market_preference1[self.step_count % 3]  += 1

        market_preference2 = np.random.normal(np.zeros(3), 0.3)
        market_preference2[(self.step_count + 1) % 3]  += 1

        # create one hot encoded action
        action_one_hot1 = [0, 0, 0]
        action_one_hot1[action1] = 1

        action_one_hot2 = [0, 0, 0]
        action_one_hot2[action2] = 1

        # reward is the dot product of the market preference and the action one hot encoded
        reward1 = np.dot(market_preference1, action_one_hot1)
        reward2 = np.dot(market_preference2, action_one_hot2)

        reward = reward1 * reward2


        self.state = (market_preference1, market_preference2)


        done = self.step_count >= self.episode_length
        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass

gymenv = SimpleGameEnv()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow_probability.python.distributions import Categorical

class PPOAgent:
    def __init__(self, env, actor_lr=0.001, critic_lr=0.001, gamma=0.99, clip_ratio=0.2, epochs=10, batch_size=64):
        self.env = env
        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)

    def build_actor(self):
        from tensorflow.keras.layers import Dropout
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=STATE_SIZE))
        # add dropout
        model.add(Dropout(0.1))
        # add dense layer
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_dim, activation='softmax'))
        return model

    def build_critic(self):
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=STATE_SIZE))
        model.add(Dense(1))
        return model

    def get_action(self, state):
        state = np.array([state])
        action_probs = self.actor.predict(state)[0]
        action_dist = Categorical(probs=action_probs)
        action = action_dist.sample()
        return int(action.numpy())

    def train(self, episodes):
        for episode in range(episodes):
            state1, state2 = self.env.reset()
            done = False
            episode_reward = 0
            states1, actions1, rewards, next_states1, dones = [], [], [], [], []
            states2 = []
            actions2 = []
            next_states2 = []
            
            while not done:
                action1 = self.get_action(state1)
                action2 = self.get_action(state2)
                action_dists1 = self.actor.predict(np.array([state1]))
                action_dists2 = self.actor.predict(np.array([state2]))

                print(f"state {state1} Action probs: {action_dists1}")
                print(f"state {state2} Action probs: {action_dists2}")
                next_statex, reward, done, _ = self.env.step(action1, action2)
                next_state1, next_state2 = next_statex
                
                states1.append(state1)
                states2.append(state2)
                actions1.append(action1)
                actions2.append(action2)
                rewards.append(reward)
                next_states1.append(next_state1)
                next_states2.append(next_state2)
                dones.append(done)
                
                state1 = next_state1
                state2 = next_state2
                episode_reward += reward
            
            # states = np.array(states)
            # states = concat np array of state1 and state2
            states = np.concatenate((np.array(states1), np.array(states2)))

            actions = np.concatenate((np.array(actions1), np.array(actions2)))
            rewards = np.concatenate((np.array(rewards), np.array(rewards)))
            next_states = np.concatenate((np.array(next_states1), np.array(next_states2)))
            dones = np.concatenate((np.array(dones), np.array(dones)))

            print('states', states.shape)
            print(states)
            print('dones', dones.shape)
            
            for _ in range(self.epochs):
                indices = np.array(range(len(states)))

                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_rewards = rewards[indices]
                batch_next_states = next_states[indices]
                batch_dones = dones[indices]
                
                batch_states = np.reshape(batch_states, (self.batch_size * 2, STATE_SIZE))
                batch_next_states = np.reshape(batch_next_states, (self.batch_size * 2, STATE_SIZE))
                
                policies = self.actor.predict(batch_states)
                values = self.critic.predict(batch_states).flatten()
                next_values = self.critic.predict(batch_next_states).flatten()
                
                returns = batch_rewards + self.gamma * next_values * (1 - batch_dones)

                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                    action_probs = self.actor(batch_states)
                    action_dist = Categorical(probs=action_probs)
                    log_probs = action_dist.log_prob(batch_actions)
                    
                    values_pred = tf.reshape(self.critic(batch_states), [-1])
                    advantages = returns - values_pred
                    
                    ratio = tf.exp(log_probs - tf.stop_gradient(log_probs))
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
                    critic_loss = tf.reduce_mean(tf.square(returns - values_pred))
                
                actor_grads = tape1.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = tape2.gradient(critic_loss, self.critic.trainable_variables)
                
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            
            # print rewards
            print('batch rewards')
            print(batch_rewards.shape)
            print(batch_rewards)
            print(f"Episode {episode + 1}: Reward = {episode_reward}")

            with open('log.txt', 'a') as f:
                f.write(f"Episode {episode + 1}: Reward = {episode_reward:.2f}\n")

env = SimpleGameEnv()
agent = PPOAgent(env, batch_size=10, gamma=0.5, actor_lr=0.001, critic_lr=0.001)

agent.train(episodes=1000)
