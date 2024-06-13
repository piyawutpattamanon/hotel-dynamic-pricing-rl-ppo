import numpy as np
import gym
from gym import spaces

class SimpleGameEnv(gym.Env):
    def __init__(self):
        super(SimpleGameEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Discrete(2)  # Our last action
        self.state = 0
        self.episode_length = 100
        self.step_count = 0
        self.last_three_actions = []

    def reset(self):
        self.state = 0
        self.step_count = 0
        self.last_three_actions = []
        return self.state

    def step(self, action):
        self.step_count += 1
        self.last_three_actions.append(action)
        if len(self.last_three_actions) > 3:
            self.last_three_actions.pop(0)
        
        opponent_action = 1 if sum(self.last_three_actions) >= 2 else 0
        reward = 1 if action != opponent_action else 0
        
        done = self.step_count >= self.episode_length
        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass

gymenv = SimpleGameEnv()


## the runnable version

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
        model.add(Dense(64, activation='relu', input_dim=1))  # Change input_dim to 1
        model.add(Dense(self.action_dim, activation='softmax'))
        return model

    def build_critic(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=1))  # Change input_dim to 1
        model.add(Dense(1))
        return model

    def get_action(self, state):
        state = np.array([[state]])  # Convert the scalar state to a 2D array with shape (1, 1)
        action_probs = self.actor.predict(state)[0]
        action_dist = Categorical(probs=action_probs)
        action = action_dist.sample()
        return int(action.numpy())  # Remove the indexing [0]
    
    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            states, actions, rewards, next_states, dones = [], [], [], [], []
            
            while not done:
                action = self.get_action(state)
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
                indices = np.random.choice(len(states), size=self.batch_size)
                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_rewards = rewards[indices]
                batch_next_states = next_states[indices]
                batch_dones = dones[indices]
                
                batch_states = np.reshape(batch_states, (self.batch_size, 1))  # Reshape to (batch_size, 1)
                batch_next_states = np.reshape(batch_next_states, (self.batch_size, 1))  # Reshape to (batch_size, 1)
                
                values = self.critic.predict(batch_states)
                next_values = self.critic.predict(batch_next_states)


                returns = batch_rewards + self.gamma * next_values * (1 - batch_dones)
                
                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                    action_probs = self.actor(batch_states)
                    action_dist = Categorical(probs=action_probs)
                    log_probs = action_dist.log_prob(batch_actions)
                    
                    values_pred = self.critic(batch_states)
                    advantages = returns - values_pred
                    
                    ratio = tf.exp(log_probs - tf.stop_gradient(log_probs))
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
                    critic_loss = tf.reduce_mean(tf.square(returns - values_pred))
                
                actor_grads = tape1.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = tape2.gradient(critic_loss, self.critic.trainable_variables)
                
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            
            print(f"Episode {episode+1}: Reward = {episode_reward}")

# Create the environment and agent
env = SimpleGameEnv()
agent = PPOAgent(env)

# Train the agent
agent.train(episodes=1000)