import matplotlib.pyplot as plt
from collections import deque
from matplotlib import style
from PIL import Image
import numpy as np
import pickle
import random
import cv2

style.use("ggplot")
import Network

SIZE = 10
SIGHT = SIZE/2
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 129  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 128  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = "MyModel.pickle"
USE_TRAINED_MODEL = True
CRIMINAL_RUN = False
USE_HYBRID_NN = True

# Environment settings
EPISODES = 500

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.997
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# For stats
ep_rewards = []
min_rewards = []
max_rewards = []
avg_rewards = []

# For more repetitive results
# random.seed(1)
# np.random.seed(1)

class cop_class:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def respawn(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def sub(self, other):
        return (self.x-other.x, self.y-other.y)

    def relative_position(self, other):
        x_val = other.x - self.x
        y_val = other.y - self.y

        if x_val > (SIZE/2):
            x_val = SIZE - x_val
        elif x_val < -((SIZE - 1)/2):
            x_val = SIZE + x_val

        if y_val > (SIZE/2):
            y_val = SIZE - y_val
        elif y_val < -((SIZE - 1)/2):
            y_val = SIZE + y_val

        if x_val > SIGHT:
            x_val = SIGHT
        elif x_val < -SIGHT:
            x_val = -SIGHT

        if y_val > SIGHT:
            y_val = SIGHT
        elif y_val < -SIGHT:
            y_val = -SIGHT

        return (x_val, y_val)

    def action(self):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if self.direction == 0:
            self.move(x=0, y=-1)
        elif self.direction == 1:
            self.move(x=0, y=1)
        elif self.direction == 2:
            self.move(x=-1, y=0)
        elif self.direction == 3:
            self.move(x=1, y=0)

    def move(self, x=0, y=0):
        self.x = (self.x + x) % SIZE
        self.y = (self.y + y) % SIZE
    
    def get_observation(self, c, t):
        c_obs = self.relative_position(c)
        t_obs = self.relative_position(t)
        self.new_obs = [c_obs[0], c_obs[1], t_obs[0], t_obs[1]]
        return self.new_obs

    def set_DQN(self,agent):
        self.agent = agent

    def perform_action(self):
        self.obs = self.new_obs
        if np.random.random() > epsilon:
            self.direction = self.agent.get_qs(self.obs)
        else:
            self.direction = np.random.randint(0, 4)
        self.action()
        return self.direction



class thief_class:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def respawn(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def sub(self, other):
        return (self.x-other.x, self.y-other.y)

    def relative_position(self, other):
        x_val = other.x - self.x
        y_val = other.y - self.y

        if x_val > (SIZE/2):
            x_val = SIZE - x_val
        elif x_val < -((SIZE - 1)/2):
            x_val = SIZE + x_val

        if y_val > (SIZE/2):
            y_val = SIZE - y_val
        elif y_val < -((SIZE - 1)/2):
            y_val = SIZE + y_val

        if x_val > SIGHT:
            x_val = SIGHT
        elif x_val < -SIGHT:
            x_val = -SIGHT

        if y_val > SIGHT:
            y_val = SIGHT
        elif y_val < -SIGHT:
            y_val = -SIGHT
            
        return (x_val, y_val)

    def action(self):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if self.direction == 0:
            self.move(x=0, y=-1)
        elif self.direction == 1:
            self.move(x=0, y=1)
        elif self.direction == 2:
            self.move(x=-1, y=0)
        elif self.direction == 3:
            self.move(x=1, y=0)

    def move(self, x=0, y=0):
        self.x = (self.x + x) % SIZE
        self.y = (self.y + y) % SIZE
    
    def get_observation(self, c1, c2):
        obs = [0,0,0,0]
        obs[0],obs[1] = self.relative_position(c1)
        obs[2],obs[3] = self.relative_position(c2)
        self.obs = obs

    def run(self):
        obs_values = list(map(abs, self.obs))
        lowest_index = obs_values.index(min(obs_values))
        if self.obs[lowest_index] < 0:
            if lowest_index in (0,2):
                self.direction = 3
            else:
                self.direction = 1
        else:
            if lowest_index in (0,2):
                self.direction = 2
            else:
                self.direction = 0
        self.action()

class GridEnvironment:

    def __init__(self):
        self.cop1 = cop_class()
        self.cop2 = cop_class()
        self.thief = thief_class()
        self.catch_count1 = 0
        self.catch_count2 = 0

        self.MOVE_PENALTY = 1
        self.COLLISION_PENALTY = 100
        self.CATCH_REWARD = 200
        self.OBSERVATION_SPACE_VALUES = 4
        self.ACTION_SPACE_SIZE = 4

        self.COP_N = 1  
        self.THIEF_N = -1 
        self.agent_colours = {1: (255, 175, 0), -1: (0, 0, 255)}

    def reset(self):
        self.cop1.respawn()
        self.cop2.respawn()
        self.thief.respawn()

        self.episode_reward = 0
        self.cop1_caught = 0
        self.cop2_caught = 0

        self.episode_step = 0
        observation1 = self.cop1.get_observation(self.cop2, self.thief)
        observation2 = self.cop2.get_observation(self.cop1, self.thief)
        return [observation1, observation2]

    def step(self):
        self.episode_step += 1
        reward = 0
        if self.cop1.x == self.thief.x and self.cop1.y == self.thief.y:
            reward = self.CATCH_REWARD
            self.cop1_caught = self.CATCH_REWARD
            self.catch_count1 += 1 
            if self.cop2.x == self.thief.x and self.cop2.y == self.thief.y:
                self.cop2_caught = self.CATCH_REWARD
        elif self.cop2.x == self.thief.x and self.cop2.y == self.thief.y:
            reward = self.CATCH_REWARD
            self.cop2_caught = self.CATCH_REWARD
            self.catch_count2 += 1 
        elif self.cop1.x == self.cop2.x and self.cop1.y == self.cop2.y:
            reward = -self.COLLISION_PENALTY
        else:
            reward = -self.MOVE_PENALTY
        
        if SHOW_PREVIEW:
            self.render(reward)

        if CRIMINAL_RUN:
            self.thief.get_observation(self.cop1, self.cop2)
            self.thief.run()

        done = False
        if reward == self.CATCH_REWARD:
            done = True

        reward1 = reward + self.cop1_caught
        reward2 = reward + self.cop2_caught
        new_observation1 = self.cop1.get_observation(self.cop2, self.thief)
        new_observation2 = self.cop1.get_observation(self.cop2, self.thief)
        return_value_1 = [new_observation1, reward1]
        return_value_2 = [new_observation2, reward2]

        return [return_value_1, return_value_2, done]

    def render(self, reward):
        grid = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        grid[self.thief.x][self.thief.y] = self.agent_colours[self.THIEF_N]
        grid[self.cop1.x][self.cop1.y] = self.agent_colours[self.COP_N] 
        grid[self.cop2.x][self.cop2.y] = self.agent_colours[self.COP_N] 
        img = Image.fromarray(grid, 'RGB')
        img = img.resize((300, 300))
        cv2.imshow("image", np.array(img))
        if reward == self.CATCH_REWARD or reward == -self.COLLISION_PENALTY:
            cv2.waitKey(500)
        else:
            cv2.waitKey(100)

env = GridEnvironment()

# Agent class
class DQNAgent:
    def __init__(self):
        if USE_TRAINED_MODEL:
            with open(MODEL_NAME, "rb") as file: 
                self.model = pickle.load(file)
            with open(MODEL_NAME, "rb") as file: 
                self.target_model = pickle.load(file)
        else:
            # Main model
            self.model = self.create_model()
            # Target network
            self.target_model = self.create_model()
            self.target_model.copy_weights(self.model)

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        learningRate =  0.001
        print("\n\nStart")
        model = Network.Network(env.OBSERVATION_SPACE_VALUES, learningRate)
        print("Input layer created")
        if USE_HYBRID_NN:
            model.addlayer(8, "relu", True)
            print("Hidden layer created")
            model.addlayer(6, "relu", True)
        else:
            model.addlayer(8, "relu")
            print("Hidden layer created")
            model.addlayer(6, "relu")
        print("Hidden layer created")
        model.addlayer(env.ACTION_SPACE_SIZE, "relu")
        print("All layers created\n\n")
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, is_done):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        # minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        minibatch = self.replay_memory

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = []
        for current_state in current_states:
            current_qs_list.append(self.model.predict(current_state))

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = []
        for new_current_state in new_current_states:
            future_qs_list.append(self.target_model.predict(new_current_state))

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(list(current_qs))

        # Fit on all samples as one batch
        for i in range(len(X)):
            self.model.train(X[i],y[i])
            
        # Update target network counter every episode
        if is_done:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.copy_weights(self.model)
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        Q_values = list(self.model.predict(state))
        return Q_values.index(max(Q_values)) 

agent = DQNAgent()
env.cop1.set_DQN(agent)
env.cop2.set_DQN(agent)

# Iterate over episodes
for episode in range(1, EPISODES + 1):
    print(episode, ":", end = " ")
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state1, current_state2 = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        action1 = env.cop1.perform_action()
        action2 = env.cop2.perform_action()
        
        return_value_1, return_value_2, done = env.step()
        new_observation1, reward1 = return_value_1[:]
        new_observation2, reward2 = return_value_2[:]
        
        # Transform new continous state to new discrete state and count reward
        if reward1 <= reward2:
            episode_reward += reward1
        else:
            episode_reward += reward2


        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state1, action1, reward1, new_observation1, done))
        agent.update_replay_memory((current_state2, action2, reward2, new_observation2, done))
        agent.train(done)

        current_state1 = new_observation1
        current_state2 = new_observation2
        step += 1

    print(episode_reward)
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)

    # if episode % 100 == 0:
    #     with open(f"./MyModel{episode//100}.pickle", "wb") as file:
    #         pickle.dump(agent.model, file)

    avg_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
    min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
    max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

    min_rewards.append(min_reward)
    max_rewards.append(max_reward)
    avg_rewards.append(avg_reward)

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

plt.plot([i for i in range(len(min_rewards))], min_rewards, label = "min")
plt.plot([i for i in range(len(max_rewards))], max_rewards, label = "max")
plt.plot([i for i in range(len(avg_rewards))], avg_rewards, label = "avg")
leg = plt.legend(loc='best', ncol=2, shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()

with open(MODEL_NAME, "wb") as file:
    pickle.dump(agent.model, file)

catch_count_total = (env.catch_count1 + env.catch_count2 + 0.0001)
print("Catching Percentage : Cop 1 :", env.catch_count1 * 100 / catch_count_total)
print("Catching Percentage : Cop 2 :", env.catch_count2 * 100 / catch_count_total)
print("Catching Percentage : Total :", catch_count_total * 100 / EPISODES)
print("Average Steps To Catch      :", 200 - (sum(ep_rewards)/len(ep_rewards)))