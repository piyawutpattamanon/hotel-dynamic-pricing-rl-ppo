import numpy as np
import gym
from gym import spaces
import tensorflow as tf

REWARD_CAP = 10
STATE_SIZE = 3 + REWARD_CAP

class SimpleGameEnv(gym.Env):
    def __init__(self):
        super(SimpleGameEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Discrete(2)  # Our last action
        self.state = 0
        self.episode_length = 20
        self.step_count = 0
        self.last_three_actions = []
        self.accumulated_reward = 0
        self.historical_rewards = []

    def reset(self):
        # initialize state as tensor [0,0,0]
        self.state = np.zeros(STATE_SIZE)
        self.step_count = 0
        self.last_three_actions = []
        self.accumulated_reward = 0
        return self.state

    def step(self, action):
        self.step_count += 1
        self.last_three_actions.append(action)
        if len(self.last_three_actions) > 3:
            self.last_three_actions.pop(0)
        
        opponent_action = 1 if sum(self.last_three_actions) >= 2 else 0
        reward = 1 if action != opponent_action else 0
        self.accumulated_reward += reward
        self.historical_rewards.append(reward)

        capped_accumulated_reward = min(self.accumulated_reward, REWARD_CAP)
        # one hot encode the accumulated reward
        reward_state = [0 for i in range(REWARD_CAP)]
        reward_state[capped_accumulated_reward-1] = 1

        force_last_three_actions = [0,0,0] + self.last_three_actions
        self.state = np.array(force_last_three_actions[-3:] + reward_state)
        # self.state = np.array(force_last_three_actions[-3:] + [reward])
        
        done = self.step_count >= self.episode_length
        return self.state, self.accumulated_reward, done, {}

    def render(self, mode='human'):
        pass

gymenv = SimpleGameEnv()


## the runnable version

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
        # model.add(Dropout(0.2))
        model.add(Dense(self.action_dim, activation='softmax'))
        return model

    def build_critic(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=STATE_SIZE))
        model.add(Dropout(0.2))
        # model.add(Dense(4, activation='relu', input_dim=STATE_SIZE, kernel_regularizer='l1'))
        model.add(Dense(1))
        # model.add(Dense(1, activation='sigmoid'))
        return model

    def get_action(self, state):
        state = np.array([state])  # Convert the scalar state to a 2D array with shape (1, 1)
        # predict but make it silent
        # import os
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
        action_probs = self.actor.predict(state)[0]
        action_dist = Categorical(probs=action_probs)
        print(f"state {state} Action probs: {action_probs}")
        action = action_dist.sample()
        return int(action.numpy())  # Remove the indexing [0]
    
    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            states, actions, rewards, next_states, dones = [], [], [], [], []
            
            while not done:
                before_state = state
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                print(f"Before state: {before_state}, Action: {action}, Next state: {next_state}, Reward: {reward}")
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                state = next_state
                episode_reward += reward

            print('=====================')
            print('end of episode', episode)
            print('total reward', episode_reward)
            print('=====================')
            
            states = np.array(states)


            # print('states', states)
            actions = np.array(actions)
            # print('actions', actions)
            rewards = np.array(rewards)
            # print('rewards', rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)
            
            for _ in range(self.epochs):
                # indices = np.random.choice(len(states), size=self.batch_size)
                indices = np.array(range(len(states)))

                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_rewards = rewards[indices]
                batch_next_states = next_states[indices]
                batch_dones = dones[indices] # booleans
                
                # issue here # solved
                batch_states = np.reshape(batch_states, (self.batch_size, STATE_SIZE))  # Reshape to (batch_size, 1)
                batch_next_states = np.reshape(batch_next_states, (self.batch_size, STATE_SIZE))  # Reshape to (batch_size, 1)
                
                # now issue here
                policies = self.actor.predict(batch_states)
                values = self.critic.predict(batch_states).flatten()
                next_values = self.critic.predict(batch_next_states).flatten()


                # TODO: Calculate returns

                returns = batch_rewards + self.gamma * next_values * (1 - batch_dones)
                # returns = batch_rewards # + next_values * self.gamma

                print('batch states')
                print(batch_states.shape)
                print(batch_states)

                print('batch values')
                print(values.shape)
                print(values)

                print('batch policies')
                print(policies.shape)
                print(policies)

                print('batch actions')
                print(batch_actions.shape)
                print(batch_actions)

                print('batch next states')
                print(batch_next_states.shape)
                print(batch_next_states)

                print('batch next_values')
                print(next_values.shape)
                print(next_values)

                print('batch rewards')
                print(batch_rewards.shape)
                print(batch_rewards)

                print('batch dones')
                print(batch_dones.shape)
                print(batch_dones)


                print('returns')
                print(returns.shape)
                print(returns)
                # input('press enter')
                
                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                    action_probs = self.actor(batch_states)
                    action_dist = Categorical(probs=action_probs)
                    print('action dist')
                    print(action_dist)

                    log_probs = action_dist.log_prob(batch_actions)
                    
                    values_pred = tf.reshape(self.critic(batch_states), [-1])
                    print('values pred')
                    print(values_pred.shape)
                    print(values_pred)
                    advantages = returns - values_pred
                    print('advantages')
                    print(advantages.shape)
                    print(advantages)
                    
                    ratio = tf.exp(log_probs - tf.stop_gradient(log_probs))
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
                    critic_loss = tf.reduce_mean(tf.square(returns - values_pred))
                
                actor_grads = tape1.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = tape2.gradient(critic_loss, self.critic.trainable_variables)
                
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            
            print(f"Episode {episode+1}: Reward = {episode_reward}")

            # append log to a file about the timestamp episode and reward
            with open('log.txt', 'a') as f:
                f.write(f"Episode {episode+1}: Reward = {episode_reward}\n")

# Create the environment and agent
env = SimpleGameEnv()
agent = PPOAgent(
    env,
    batch_size=20,
    gamma=0.99,
    actor_lr=0.001,
    critic_lr=0.001,
)

# Train the agent
agent.train(episodes=1000)