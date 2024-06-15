import numpy as np
import gym
from gym import spaces
import tensorflow as tf

REWARD_CAP = 0
STATE_SIZE = 9 + REWARD_CAP

class SimpleGameEnv(gym.Env):
    def __init__(self):
        super(SimpleGameEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: rock, 1: paper, 2: scissors
        self.observation_space = spaces.Discrete(3)  # Our last action
        self.state = 0
        self.episode_length = 100
        self.step_count = 0
        self.last_three_actions = []
        self.accumulated_reward = 0
        self.historical_rewards = []

    def reset(self):
        self.state = np.zeros(STATE_SIZE)
        self.step_count = 0
        self.last_three_actions = []
        self.accumulated_reward = 0
        return self.state

    def step(self, action):
        self.step_count += 1

        force_last_three_actions = ([0, 0, 0] + self.last_three_actions)[-3:]
        opponent_action = (sorted(force_last_three_actions)[1] + 1) % 3
        # opponent_action = np.random.choice([0, 1, 2])  # Random opponent action

        
        
        # Determine reward
        if action == opponent_action:
            reward = 0  # Draw
        elif (action == 0 and opponent_action == 2) or \
             (action == 1 and opponent_action == 0) or \
             (action == 2 and opponent_action == 1):
            reward = 1  # Win
        else:
            reward = -1  # Lose

        print(f"force_last_three_actions: {force_last_three_actions}")
        print(f"Opponent action: {opponent_action} Your action: {action} reward: {reward}")
        
        self.accumulated_reward += reward
        self.historical_rewards.append(reward)

        capped_accumulated_reward = min(self.accumulated_reward, REWARD_CAP)
        reward_state = [0 for i in range(REWARD_CAP)]
        if REWARD_CAP > 0:
            reward_state[capped_accumulated_reward - 1] = 1

        self.last_three_actions.append(action)
        if len(self.last_three_actions) > 3:
            self.last_three_actions.pop(0)

        force_last_three_actions = [0, 0, 0] + self.last_three_actions
        next_state_raw = np.array(force_last_three_actions[-3:])

        # make it one hot encoding of last three actions (rock, paper, scissors)
        # for example if last three actions are [0, 1, 2] then state will be [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        next_state_one_hot = np.eye(3)[next_state_raw].flatten()
        self.state = next_state_one_hot


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
        model = Sequential()
        model.add(Dense(3, activation='relu', input_dim=STATE_SIZE))
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
            state = self.env.reset()
            done = False
            episode_reward = 0
            states, actions, rewards, next_states, dones = [], [], [], [], []
            
            while not done:
                before_state = state
                action = self.get_action(state)
                action_dists = self.actor.predict(np.array([state]))
                print(f"state {state} Action probs: {action_dists}")
                next_state, reward, done, _ = self.env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                state = next_state
                episode_reward += reward
            
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)
            
            for _ in range(self.epochs):
                indices = np.array(range(len(states)))

                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_rewards = rewards[indices]
                batch_next_states = next_states[indices]
                batch_dones = dones[indices]
                
                batch_states = np.reshape(batch_states, (self.batch_size, STATE_SIZE))
                batch_next_states = np.reshape(batch_next_states, (self.batch_size, STATE_SIZE))
                
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
                f.write(f"Episode {episode + 1}: Reward = {episode_reward}\n")

env = SimpleGameEnv()
agent = PPOAgent(env, batch_size=100, gamma=0.5, actor_lr=0.001, critic_lr=0.001)

agent.train(episodes=1000)
