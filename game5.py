import numpy as np
import gym
from gym import spaces
import tensorflow as tf
import copy

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow_probability.python.distributions import Categorical
from tensorflow.keras.layers import Dropout
from sklearn.metrics.pairwise import cosine_similarity


class HotelPricingGameEnv(gym.Env):
    def __init__(
        self,
        episode_length=20,
        num_customers=15,
        num_hotel_rooms=5,
    ):
        super(HotelPricingGameEnv, self).__init__()
        self.num_customers = num_customers
        self.num_hotel_rooms = num_hotel_rooms

        self.action_space = spaces.Discrete(2)  # 0: rock, 1: paper, 2: scissors
        self.observation_space = spaces.Discrete(5)  # Our last action
        self.state = 0
        self.episode_length = episode_length
        self.step_count = 0
        self.last_three_actions = []
        self.accumulated_reward = 0
        self.historical_rewards = []

        self.customers = [self.create_customer() for _ in range(self.num_customers)]

        self.original_hotel_rooms = [
            self.create_hotel_room() for _ in range(self.num_hotel_rooms)
        ]

    def create_customer(self):
        customer = {
            "affinity": np.random.normal(np.zeros(3), 1.0),
            "willingness_to_pay": np.random.uniform(100, 5000),
        }
        return customer

    def create_hotel_room(self):
        room = {
            "id": np.random.randint(0, 100000),
            "affinity": np.random.normal(np.zeros(3), 1.0),
            "price": np.exp(np.random.uniform(np.log(100), np.log(1000))),
            "time_vacant": 0,
        }
        return room

    def get_hotel_room_observation(self, room):
        observation = np.concatenate(
            (room["affinity"], [np.log(room["price"])], [room["time_vacant"]])
        )
        return observation

    def get_hotel_rooms_observations(self):
        observations = [
            self.get_hotel_room_observation(room) for room in self.hotel_rooms
        ]
        return observations

    def reset(self):
        self.hotel_rooms = copy.deepcopy(self.original_hotel_rooms)
        self.state = self.get_hotel_rooms_observations()
        self.step_count = 0
        self.last_three_actions = []
        self.accumulated_reward = 0

        return self.state

    def step(self, action):
        self.step_count += 1

        self.customers.sort(key=lambda x: -x["willingness_to_pay"])

        for index, room_action in enumerate(action):
            if room_action == 0:
                self.hotel_rooms[index]["price"] *= 0.7
            elif room_action == 1:
                self.hotel_rooms[index]["price"] *= 1.3

        print(
            "williness to pay",
            [customer["willingness_to_pay"] for customer in self.customers],
        )
        print("room prices", [room["price"] for room in self.hotel_rooms])

        revenue = 0
        bought_room_ids = set()
        for customer in self.customers:

            customer_affinity = customer["affinity"]
            customer_willingness_to_pay = customer["willingness_to_pay"]

            candidates = self.hotel_rooms.copy()
            sorted_candidates = sorted(
                candidates, key=lambda x: -np.dot(x["affinity"], customer_affinity)
            )
            for room in sorted_candidates:
                if room["id"] in bought_room_ids:
                    continue

                cosine_score = cosine_similarity(
                    [room["affinity"]], [customer_affinity]
                )[0][0]
                pure_price_chance = (
                    customer_willingness_to_pay / (room["price"] + 1e-6)
                ) ** 2
                buy_chance = cosine_score * pure_price_chance

                print(
                    f"cosine_score: {cosine_score} pure_price_chance: {pure_price_chance} buy_chance: {buy_chance}"
                )

                if np.random.uniform() < buy_chance:
                    revenue += room["price"]
                    bought_room_ids.add(room["id"])
                    break

        print("revenue", revenue)
        print("bought room ids", bought_room_ids)

        vacant_penalty = 0
        for room in self.hotel_rooms:
            if room["id"] in bought_room_ids:
                room["time_vacant"] = 0
            else:
                room["time_vacant"] += 1
                vacant_penalty += 100 * np.log(room["time_vacant"])

        print("vacant_penalty", vacant_penalty)

        reward = revenue - vacant_penalty

        self.state = self.get_hotel_rooms_observations()

        done = self.step_count >= self.episode_length
        return self.state, reward, done, {}

    def render(self, mode="human"):
        pass


class PPOAgent:
    def __init__(
        self,
        env,
        actor_lr=0.001,
        critic_lr=0.001,
        gamma=0.99,
        clip_ratio=0.2,
        epochs=10,
        batch_size=64,
    ):
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
        model.add(Dense(200, activation="relu", input_dim=self.env.observation_space.n))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation="relu"))

        model.add(Dropout(0.1))

        model.add(Dense(16, activation="relu"))

        # model.add(Dense(self.action_dim, activation='softmax'))

        def softmax_with_caps(x):
            # Apply softmax to the input
            probabilities = tf.nn.softmax(x)

            # Define the minimum and maximum caps
            min_cap = 0.1
            max_cap = 0.9

            # Clip the probabilities to be within the specified range
            clipped_probabilities = tf.clip_by_value(probabilities, min_cap, max_cap)

            # Normalize the clipped probabilities to ensure they sum to 1
            normalized_probabilities = clipped_probabilities / tf.reduce_sum(
                clipped_probabilities, axis=-1, keepdims=True
            )

            return normalized_probabilities

        model.add(Dense(self.action_dim))
        model.add(tf.keras.layers.Activation(softmax_with_caps))

        return model

    def build_critic(self):
        model = Sequential()
        model.add(Dense(100, activation="relu", input_dim=self.env.observation_space.n))
        model.add(Dropout(0.1))
        model.add(Dense(16, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        return model

    def get_action(self, state):
        print(f"state {state}")
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
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

            while not done:
                action = [self.get_action(observation) for observation in state]
                action_dist = [
                    self.actor.predict(np.array([observation])) for observation in state
                ]

                print(f"Action dist: {action_dist}")
                print(f"Action: {action}")
                next_state, reward, done, _ = self.env.step(action)

                reward /= 1000

                states += state
                actions += action
                rewards += [reward for _ in range(len(state))]
                next_states += next_state
                dones += [done for _ in range(len(state))]

                state = next_state
                episode_reward += reward

            # transform states, actions, rewards, next_states, dones into np arrays
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

                batch_states = np.reshape(
                    batch_states,
                    (
                        batch_states.size // self.env.observation_space.n,
                        self.env.observation_space.n,
                    ),
                )
                batch_next_states = np.reshape(
                    batch_next_states,
                    (
                        batch_states.size // self.env.observation_space.n,
                        self.env.observation_space.n,
                    ),
                )

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
                    clipped_ratio = tf.clip_by_value(
                        ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                    )
                    actor_loss = -tf.reduce_mean(
                        tf.minimum(ratio * advantages, clipped_ratio * advantages)
                    )
                    critic_loss = tf.reduce_mean(tf.square(returns - values_pred))

                actor_grads = tape1.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = tape2.gradient(
                    critic_loss, self.critic.trainable_variables
                )

                self.actor_optimizer.apply_gradients(
                    zip(actor_grads, self.actor.trainable_variables)
                )
                self.critic_optimizer.apply_gradients(
                    zip(critic_grads, self.critic.trainable_variables)
                )

            print(f"Episode {episode + 1}: Reward = {episode_reward}")

            with open("log.txt", "a") as f:
                f.write(f"Episode {episode + 1}: Reward = {episode_reward:,.2f}\n")


np.random.seed(0)
EPISODE_LENGTH = 20

env = HotelPricingGameEnv(
    episode_length=EPISODE_LENGTH, num_customers=15, num_hotel_rooms=5
)
agent = PPOAgent(
    env, batch_size=EPISODE_LENGTH, gamma=0.2, actor_lr=0.001, critic_lr=0.001
)

agent.train(episodes=200)
