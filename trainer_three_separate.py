import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from agents import *
from params import *

style.use("ggplot")

def train_separate(EPISODES=100):
    EPSILON = 1.0
    EPS_DECAY, SHOW_EVERY = get_params(EPISODES)

    start_q_table_1 = None
    start_q_table_2 = None
    start_q_table_3 = None
    # start_q_table_1 = "Models/qtable_1.pickle" # None or Filename
    # start_q_table_2 = "Models/qtable_2.pickle" # None or Filename
    # start_q_table_3 = "Models/qtable_3.pickle" # None or Filename

    if start_q_table_1 is None:
        sight_range = 2*SIGHT+1

        q_table_1 = np.zeros((sight_range, sight_range, sight_range, sight_range, sight_range, sight_range, 4))
        q_table_2 = np.zeros((sight_range, sight_range, sight_range, sight_range, sight_range, sight_range, 4))
        q_table_3 = np.zeros((sight_range, sight_range, sight_range, sight_range, sight_range, sight_range, 4))

        for i in range(- SIGHT, SIGHT+1):
            for ii in range(- SIGHT, SIGHT+1):
                for iii in range(- SIGHT, SIGHT+1):
                        for iiii in range(- SIGHT, SIGHT+1):
                            for iiiii in range(- SIGHT, SIGHT+1):
                                    for iiiiii in range(- SIGHT, SIGHT+1):
                                        q_table_1[i, ii, iii, iiii, iiiii, iiiiii, :] = [np.random.uniform(-1, 1) for i in range(4)]
                                        q_table_2[i, ii, iii, iiii, iiiii, iiiiii, :] = [np.random.uniform(-1, 1) for i in range(4)]
                                        q_table_3[i, ii, iii, iiii, iiiii, iiiiii, :] = [np.random.uniform(-1, 1) for i in range(4)]

    else:
        with open(start_q_table_1, "rb") as f:
            q_table_1 = pickle.load(f)
        with open(start_q_table_2, "rb") as f:
            q_table_2 = pickle.load(f)
        with open(start_q_table_3, "rb") as f:
            q_table_3 = pickle.load(f)


    episode_rewards = []
    
    cop1 = cop_class(SIZE, LEARNING_RATE, DISCOUNT)
    cop2 = cop_class(SIZE, LEARNING_RATE, DISCOUNT)
    cop3 = cop_class(SIZE, LEARNING_RATE, DISCOUNT)
    thief = thief_class()
    catch_count1 = 0
    catch_count2 = 0
    catch_count3 = 0

    for episode in range(EPISODES):
        cop1.respawn()
        cop2.respawn()
        cop3.respawn()
        thief.respawn()

        episode_reward = 0
        cop1_caught = 0
        cop2_caught = 0
        cop3_caught = 0
        cop1_collide = 0
        cop2_collide = 0
        cop3_collide = 0
        cop1.get_observation([cop2, cop3], thief)
        cop2.get_observation([cop3, cop1], thief)
        cop3.get_observation([cop1, cop2], thief)
        for i in range(MAX_STEPS):
            cop1.perform_action(q_table_1, EPSILON)
            cop2.perform_action(q_table_2, EPSILON)
            cop3.perform_action(q_table_3, EPSILON)
            if cop1.x == thief.x and cop1.y == thief.y:
                reward = CATCH_REWARD
                cop1_caught = CATCH_REWARD
                catch_count1 += 1
            elif cop2.x == thief.x and cop2.y == thief.y:
                reward = CATCH_REWARD
                cop2_caught = CATCH_REWARD
                catch_count2 += 1
            elif cop3.x == thief.x and cop3.y == thief.y:
                reward = CATCH_REWARD
                cop3_caught = CATCH_REWARD
                catch_count3 += 1
            elif cop1.x == cop2.x and cop1.y == cop2.y:
                reward = -MOVE_PENALTY
                cop1_collide = -COLLISION_PENALTY
                cop2_collide = -COLLISION_PENALTY
                if cop1.x == cop3.x and cop1.y == cop3.y:
                    cop3_collide = -COLLISION_PENALTY
            elif cop1.x == cop3.x and cop1.y == cop3.y:
                reward = -MOVE_PENALTY
                cop1_collide = -COLLISION_PENALTY
                cop3_collide = -COLLISION_PENALTY
            elif cop2.x == cop3.x and cop2.y == cop3.y:
                reward = -MOVE_PENALTY
                cop2_collide = -COLLISION_PENALTY
                cop3_collide = -COLLISION_PENALTY
            else:
                reward = -MOVE_PENALTY
            
            if SHOW:
                env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
                env[thief.x][thief.y] = agent_colours[THIEF_N]
                env[cop1.x][cop1.y] = agent_colours[COP_N]
                env[cop2.x][cop2.y] = agent_colours[COP_N]
                env[cop3.x][cop3.y] = agent_colours[COP_N]
                img = Image.fromarray(env, 'RGB')
                img = img.resize((300, 300))
                cv2.imshow("image", np.array(img)) 
                if reward == CATCH_REWARD:
                    if cv2.waitKey(500) & 0xFF == ord('q'):
                        break
                else:
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break

            thief.get_observation([cop1, cop2, cop3])
            thief.run()

            cop1.update_table(q_table_1, reward + cop1_caught + cop1_collide , [cop2, cop3], thief)
            cop2.update_table(q_table_2, reward + cop2_caught + cop2_collide , [cop3, cop1], thief)
            cop3.update_table(q_table_3, reward + cop3_caught + cop3_collide , [cop1, cop2], thief)

            episode_reward += reward
            if reward == CATCH_REWARD:
                break

        if PRINT_EPISODE:
            print(episode + 1, ": STEPS :", MAX_STEPS - episode_reward)
        episode_rewards.append(episode_reward)
        EPSILON  *= EPS_DECAY

    moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

    plt.plot([i+SHOW_EVERY for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f'Rewards')
    plt.xlabel('episode #')
    plt.savefig('Graphs/separate_performance.png')
    
    if SHOW_PLOT:
        plt.show()

    with open(f'Models/qtable_1.pickle', 'wb') as f:
        pickle.dump(q_table_1, f)
    with open(f'Models/qtable_2.pickle', 'wb') as f:
        pickle.dump(q_table_2, f)
    with open(f'Models/qtable_3.pickle', 'wb') as f:
        pickle.dump(q_table_3, f)

    catch_count_total = (catch_count1 + catch_count2 + catch_count3)
    print("Catching Percentage : Cop 1  :", catch_count1 * 100 / catch_count_total)
    print("Catching Percentage : Cop 2  :", catch_count2 * 100 / catch_count_total)
    print("Catching Percentage : Cop 3  :", catch_count3 * 100 / catch_count_total)
    print("Catching Percentage : Total  :", catch_count_total * 100 / EPISODES)
    print("Average Reward               :", sum(episode_rewards)/len(episode_rewards))
    print("Average Steps To Catch       :", MAX_STEPS - (sum(episode_rewards)/len(episode_rewards)))

if __name__=="__main__":
    train_separate(EPISODES=1)