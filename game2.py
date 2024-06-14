import numpy as np
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the environment
REWARD_CAP = 0
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
        self.state = np.zeros(STATE_SIZE)
        self.step_count = 0
        self.last_three_actions = []
        self.accumulated_reward = 0
        return self.state

    def step(self, action):
        self.step_count += 1

        opponent_action = 1 if sum(self.last_three_actions) >= 2 else 0
        
        reward = 1 if action != opponent_action else -1
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
        self.state = np.array(force_last_three_actions[-3:])
        
        done = self.step_count >= self.episode_length
        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass

# Define the SAC agent
class SACAgent:
    def __init__(self, env, actor_lr=0.001, critic_lr=0.001, alpha_lr=0.001, gamma=0.99, tau=0.005, batch_size=64, memory_capacity=100000):
        self.env = env
        self.state_dim = STATE_SIZE
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.target_entropy = -np.prod(env.action_space.shape).item()  # add this line


        self.actor = self.build_actor()
        self.critic1 = self.build_critic()
        self.critic2 = self.build_critic()
        self.target_critic1 = self.build_critic()
        self.target_critic2 = self.build_critic()
        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic1_optimizer = Adam(learning_rate=critic_lr)
        self.critic2_optimizer = Adam(learning_rate=critic_lr)
        self.alpha = tf.Variable(0.2, dtype=tf.float32)
        self.alpha_optimizer = Adam(learning_rate=alpha_lr)
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)

        self.memory = []
        self.memory_capacity = memory_capacity

    def build_actor(self):
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=self.state_dim))
        model.add(Dense(self.action_dim, activation='softmax'))
        return model

    def build_critic(self):
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=self.state_dim + self.action_dim))
        model.add(Dense(1))
        return model

    def get_action(self, state):
        state = np.array([state])
        action_probs = self.actor(state).numpy()[0]
        action = np.random.choice(self.action_dim, p=action_probs)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        indices = np.random.choice(len(self.memory), size=self.batch_size)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.store_transition(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if len(self.memory) >= self.batch_size:
                    self.update_model()

            print(f"Episode {episode + 1}: Reward = {episode_reward}")

    def update_model(self):
        states, actions, rewards, next_states, dones = self.sample_memory()

        # Update Critic networks
        next_actions = np.array([self.get_action(next_state) for next_state in next_states])
        next_actions_one_hot = tf.one_hot(next_actions, self.action_dim)
        
        # Calculate target Q values
        next_q1 = self.target_critic1(tf.concat([next_states, next_actions_one_hot], axis=1))
        next_q2 = self.target_critic2(tf.concat([next_states, next_actions_one_hot], axis=1))
        
        # Log probability of the next action
        next_action_probs = self.actor(next_states)
        next_action_dist = tf.compat.v1.distributions.Categorical(probs=next_action_probs)
        next_log_probs = next_action_dist.log_prob(next_actions)
        
        # Calculate the target Q value
        next_q = tf.minimum(next_q1, next_q2) - self.alpha * next_log_probs
        q_target = rewards + self.gamma * (1 - dones) * next_q

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            actions_one_hot = tf.one_hot(actions, self.action_dim)
            q1 = self.critic1(tf.concat([states, actions_one_hot], axis=1))
            q2 = self.critic2(tf.concat([states, actions_one_hot], axis=1))
            critic1_loss = tf.reduce_mean(tf.square(q_target - q1))
            critic2_loss = tf.reduce_mean(tf.square(q_target - q2))

        critic1_grads = tape1.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_grads = tape2.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))

        # Update Actor network
        with tf.GradientTape() as tape:
            actions_probs = self.actor(states)
            actions_dist = tf.compat.v1.distributions.Categorical(probs=actions_probs)
            log_probs = actions_dist.log_prob(actions)
            q1 = self.critic1(tf.concat([states, actions_probs], axis=1))
            q2 = self.critic2(tf.concat([states, actions_probs], axis=1))
            q_values = tf.minimum(q1, q2)
            actor_loss = tf.reduce_mean(self.alpha * log_probs - q_values)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update alpha
        with tf.GradientTape() as tape:
            alpha_loss = -tf.reduce_mean(self.alpha * (log_probs + self.target_entropy))

        alpha_grads = tape.gradient(alpha_loss, [self.alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.alpha]))

        # Soft update the target networks
        self.update_target_network(self.critic1, self.target_critic1)
        self.update_target_network(self.critic2, self.target_critic2)

    def update_target_network(self, source, target):
        for src_var, tgt_var in zip(source.variables, target.variables):
            tgt_var.assign(self.tau * src_var + (1 - self.tau) * tgt_var)


# Create the environment and agent
env = SimpleGameEnv()
agent = SACAgent(
    env,
    batch_size=20,
    gamma=0.5,
    actor_lr=0.001,
    critic_lr=0.001,
)

# Train the agent
agent.train(episodes=1000)
