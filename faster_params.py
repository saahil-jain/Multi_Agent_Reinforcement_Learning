from numba import jit, int64, float64

SHOW=False
PRINT_EPISODE = False
SHOW_PLOT = False

# Map
SIZE = 10
SIGHT = 3
MAX_STEPS = 400

# Rewards
MOVE_PENALTY = 1
COLLISION_PENALTY = 100
CATCH_REWARD = MAX_STEPS

# Train Parameters

LEARNING_RATE = 0.1
DISCOUNT = 0.95

# Visualizations
COP_N = 1  
THIEF_N = -1 
agent_colours = {1: (255, 175, 0), -1: (0, 0, 255)}

@jit
def get_params(episodes):
    # EPISODES = 10_000
    # EPS_DECAY = 0.9996
    # SHOW_EVERY = 1000
    if episodes<10_000:
        return 0.9996, 1000

    # EPISODES = 1_00_000
    # EPS_DECAY = 0.99996
    # SHOW_EVERY = 10_000
    if episodes<1_00_000:
        return 0.99996, 10_000

    # EPISODES = 10_00_000
    # EPS_DECAY = 0.9999965
    # SHOW_EVERY = 1_00_000
    if episodes<10_00_000:
        return 0.9999965, 1_00_000