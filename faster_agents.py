import numpy as np
from params import *
from numba import jit, int32, int64, float64#, typeof
from numba.experimental import jitclass


cop_spec = [
    ('SIZE', int32),
    ('learning_rate', float64),
    ('discount', float64),
    ('x', int32),
    ('y', int32),
    ('obs', int64[:]),
    ('new_obs', int64[:]),
    ('direction', int32),
]

@jitclass(cop_spec)
class fast_cop_class:
    def __init__(self, SIZE=10, learning_rate=1, discount=0.95):
        self.SIZE = SIZE
        self.x = np.random.randint(0, self.SIZE)
        self.y = np.random.randint(0, self.SIZE)

        self.learning_rate = learning_rate
        self.discount = discount

    def respawn(self):
        self.x = np.random.randint(0, self.SIZE)
        self.y = np.random.randint(0, self.SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def sub(self, other):
        return (self.x-other.x, self.y-other.y)

    def relative_position(self, other):
        x_val = 0
        if self.x > other.x:
            distance1 = self.x - other.x
            distance2 = other.x + self.SIZE - self.x
            if distance2 < distance1:
                x_val = -distance2
            else:
                x_val = distance1
        elif self.x < other.x:
            distance1 = other.x - self.x
            distance2 = self.x+self.SIZE - other.x
            if distance2 < distance1:
                x_val = distance2
            else:
                x_val = -distance1
        
        y_val = 0
        if self.y > other.y:
            distance1 = self.y - other.y
            distance2 = other.y+self.SIZE - self.y
            if distance2 < distance1:
                y_val = - distance2
            else:
                y_val = distance1
        elif self.y < other.y:
            distance1 = other.y - self.y
            distance2 = self.y + self.SIZE - other.y
            if distance2 < distance1:
                y_val = distance2
            else:
                y_val = -distance1
                
        x_val = max(x_val, -SIGHT)
        x_val = min(x_val, SIGHT)
        y_val = max(y_val, -SIGHT)
        y_val = min(y_val, SIGHT)
        return [x_val, y_val]

    def action(self):
        if self.direction == 0:
            self.move(x=0, y=-1)
        elif self.direction == 1:
            self.move(x=0, y=1)
        elif self.direction == 2:
            self.move(x=-1, y=0)
        elif self.direction == 3:
            self.move(x=1, y=0)

    def move(self, x=0, y=0):
        self.x = (self.x + x) % self.SIZE
        self.y = (self.y + y) % self.SIZE
    
    def get_observation(self, cops, thief):
        self.new_obs = np.zeros(6, dtype=int64)
        for index, cop in enumerate(cops):
            self.new_obs[index*2], self.new_obs[index*2+1] = self.relative_position(cop)
        self.new_obs[-2], self.new_obs[-2] = self.relative_position(thief)

    def perform_action(self, q_table, epsilon):
        self.obs = self.new_obs
        if np.random.random() > epsilon:
            direction_q_values = q_table[self.obs[0]][self.obs[1]][self.obs[2]][self.obs[3]][self.obs[4]][self.obs[5]]
            self.direction = np.argmax(direction_q_values)
        else:
            self.direction = np.random.randint(0, 4)
        self.action()

    def update_table(self, q_table, reward, cops, thief):
        self.get_observation(cops, thief)
        max_future_q = np.max(q_table[self.new_obs[0]][self.new_obs[1]][self.new_obs[2]][self.new_obs[3]][self.new_obs[4]][self.new_obs[5]])
        current_q = q_table[self.obs[0]][self.obs[1]][self.obs[2]][self.obs[3]][self.obs[4]][self.obs[5]][self.direction]
        if reward >= 500:
            new_q = 500
        else:
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount * max_future_q)
        q_table[self.obs[0]][self.obs[1]][self.obs[2]][self.obs[3]][self.obs[4]][self.obs[5]][self.direction] = new_q

thief_spec = [
    ('SIZE', int32),
    ('x', int32),
    ('y', int32),
    ('obs', int64[:]),
    ('direction', int32),
    ('closest_cop', int32)
]

@jitclass(thief_spec)
class fast_thief_class:
    def __init__(self, SIZE = 10):
        self.SIZE = SIZE
        self.x = np.random.randint(0, self.SIZE)
        self.y = np.random.randint(0, self.SIZE)

    def respawn(self):
        self.x = np.random.randint(0, self.SIZE)
        self.y = np.random.randint(0, self.SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def sub(self, other):
        return (self.x-other.x, self.y-other.y)

    def relative_position(self, other):
        x_val = 0
        if self.x > other.x:
            distance1 = self.x - other.x
            distance2 = other.x + self.SIZE - self.x
            if distance2 < distance1:
                x_val = -distance2
            else:
                x_val = distance1
        elif self.x < other.x:
            distance1 = other.x - self.x
            distance2 = self.x+self.SIZE - other.x
            if distance2 < distance1:
                x_val = distance2
            else:
                x_val = -distance1
        
        y_val = 0
        if self.y > other.y:
            distance1 = self.y - other.y
            distance2 = other.y+self.SIZE - self.y
            if distance2 < distance1:
                y_val = - distance2
            else:
                y_val = distance1
        elif self.y < other.y:
            distance1 = other.y - self.y
            distance2 = self.y + self.SIZE - other.y
            if distance2 < distance1:
                y_val = distance2
            else:
                y_val = -distance1

        return [x_val, y_val]

    def get_away_direction(self, distance):
        directions = []
        if abs(distance[0]) < abs(distance[1]):
            if distance[0] < 0:
                directions.append(2)
            elif distance[0] > 0:
                directions.append(3)
            else:
                directions.append(2)
                directions.append(3)

            if distance[1] < 0:
                directions.append(1)
            elif distance[1] > 0:
                directions.append(0)
            else:
                directions.append(1)
                directions.append(0)
        
        else:
            if distance[1] < 0:
                directions.append(1)
            elif distance[1] > 0:
                directions.append(0)
            else:
                directions.append(1)
                directions.append(0)

            if distance[0] < 0:
                directions.append(2)
            elif distance[0] > 0:
                directions.append(3)
            else:
                directions.append(2)
                directions.append(3)

        return directions

    def action(self):
        if self.direction == 0:
            self.move(x=0, y=-1)
        elif self.direction == 1:
            self.move(x=0, y=1)
        elif self.direction == 2:
            self.move(x=-1, y=0)
        elif self.direction == 3:
            self.move(x=1, y=0)

    def move(self, x=0, y=0):
        self.x = (self.x + x) % self.SIZE
        self.y = (self.y + y) % self.SIZE
    
    def get_observation(self, cops):
        self.obs = np.zeros(6, dtype=int64)
        for index, cop in enumerate(cops):
            self.obs[index*2], self.obs[index*2+1] = self.relative_position(cop)

    def run(self):

        cop_positions = [self.obs[0:2], self.obs[2:4], self.obs[4:6]]

        run_directions = []
        for cop_position in cop_positions:
            run_directions.append(self.get_away_direction(cop_position))

        possible_escape_all = set(run_directions[0])
        for run_direction in run_directions:
            possible_escape_all = possible_escape_all.intersection(set(run_direction))

        if len(possible_escape_all)==0:
            distance_from_cops = []
            for cop_position in cop_positions:
                distance_from_cops.append(abs(cop_position[0]) + abs(cop_position[1]))

            closest_cop = 0
            min_distance = distance_from_cops[0]
            for i in range(1, len(distance_from_cops)):
                if distance_from_cops[i] < min_distance:
                    min_distance = distance_from_cops[i]
                    closest_cop = i

            self.direction = run_directions[closest_cop][0]
        else:
            self.direction = list(possible_escape_all)[0]
        self.action()
