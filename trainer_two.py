import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 10
SIGHT = 5
EPISODES = 5
MOVE_PENALTY = 1
COLLISION_PENALTY = 100
CATCH_REWARD = 200
epsilon = 0.0
EPS_DECAY = 0.9998
SHOW_EVERY = 1
show = True

start_q_table_1 = "Models/two_cop1_qtable.pickle" # None or Filename
start_q_table_2 = "Models/two_cop2_qtable.pickle" # None or Filename

def truncate(n):
    return int((n * 1000) / 1000)
if start_q_table_1 is None:
    # initialize the q-table#
    q_table_1 = {}
    q_table_2 = {}
    for i in range(- SIGHT, SIGHT+1):
        for ii in range(- SIGHT, SIGHT+1):
            for iii in range(- SIGHT, SIGHT+1):
                    for iiii in range(- SIGHT, SIGHT+1):
                        q_table_1[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(4)]
                        q_table_2[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(4)]

else:
    with open(start_q_table_1, "rb") as f:
        q_table_1 = pickle.load(f)
    with open(start_q_table_2, "rb") as f:
        q_table_2 = pickle.load(f)

LEARNING_RATE = 0.1
DISCOUNT = 0.95

COP_N = 1  
THIEF_N = -1 
agent_colours = {1: (255, 175, 0), -1: (0, 0, 255)}

class cop_class:
    def __init__(self, q_table):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
        self.q_table  = q_table

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
        self.new_obs = (c_obs, t_obs)
        return self.new_obs

    def perform_action(self):
        self.obs = self.new_obs
        if np.random.random() > epsilon:
            self.direction = np.argmax(self.q_table[self.obs])
        else:
            self.direction = np.random.randint(0, 4)
        self.action()

    def update_table(self, reward, c, t):
        self.get_observation(c, t)
        max_future_q = np.max(self.q_table[self.new_obs])
        current_q = self.q_table[self.obs][self.direction]
        if reward == CATCH_REWARD*2:
            new_q = CATCH_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        self.q_table[self.obs][self.direction] = new_q


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

episode_rewards = []
cop1 = cop_class(q_table_1)
cop2 = cop_class(q_table_2)
thief = thief_class()
catch_count1 = 0
catch_count2 = 0

for episode in range(EPISODES):
    cop1.respawn()
    cop2.respawn()
    thief.respawn()

    episode_reward = 0
    cop1_caught = 0
    cop2_caught = 0
    cop1.get_observation(cop2, thief)
    cop2.get_observation(cop1, thief)
    for i in range(200):
        cop1.perform_action()
        cop2.perform_action()
        if cop1.x == thief.x and cop1.y == thief.y:
            reward = CATCH_REWARD
            cop1_caught = CATCH_REWARD
            catch_count1 += 1
            if cop2.x == thief.x and cop2.y == thief.y:
                cop2_caught = CATCH_REWARD
        elif cop2.x == thief.x and cop2.y == thief.y:
            reward = CATCH_REWARD
            cop2_caught = CATCH_REWARD
            catch_count2 += 1
        elif cop1.x == cop2.x and cop1.y == cop2.y:
            reward = -COLLISION_PENALTY
        else:
            reward = -MOVE_PENALTY
        
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[thief.x][thief.y] = agent_colours[THIEF_N]
            env[cop1.x][cop1.y] = agent_colours[COP_N] 
            env[cop2.x][cop2.y] = agent_colours[COP_N] 
            img = Image.fromarray(env, 'RGB')
            img = img.resize((300, 300))
            cv2.imshow("image", np.array(img)) 
            if reward == CATCH_REWARD or reward == -COLLISION_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

        thief.get_observation(cop1, cop2)
        thief.run()

        cop1.update_table(reward+cop1_caught , cop2, thief)
        cop2.update_table(reward+cop2_caught , cop1, thief)

        episode_reward += reward
        if reward == CATCH_REWARD or reward == -COLLISION_PENALTY:
            break

    print(episode + 1, ": STEPS :", 200 - episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

# plt.plot([i for i in range(len(moving_avg))], moving_avg)
# plt.plot([i for i in range(len(episode_rewards))], episode_rewards)
# plt.ylabel(f"Reward {SHOW_EVERY}ma")
# plt.xlabel("episode #")
# plt.show()

# with open(f"Models/two_cop1_qtable.pickle", "wb") as f:
#     pickle.dump(cop1.q_table, f)
# with open(f"Models/two_cop2_qtable.pickle", "wb") as f:
#     pickle.dump(cop2.q_table, f)
catch_count_total = (catch_count1 + catch_count2)
print("Catching Percentage : Cop 1 :", catch_count1 * 100 / catch_count_total)
print("Catching Percentage : Cop 2 :", catch_count2 * 100 / catch_count_total)
print("Catching Percentage : Total :", catch_count_total * 100 / EPISODES)
print("Average Steps To Catch      :", 200 - (sum(episode_rewards)/len(episode_rewards)))