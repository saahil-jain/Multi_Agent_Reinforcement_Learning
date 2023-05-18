import os
from trainer_three_separate import *
from trainer_three_common import *
import time

print("Using Same Q-tables :")
start_time = time.time()
train_common(EPISODES=100)
end_time = time.time()
execution_time = end_time - start_time
print("Execution Time               :", execution_time, "seconds")

# TODO
# PLT . clear 

print("\nUsing Different Q-tables :")
start_time = time.time()
train_separate(EPISODES=100)
end_time = time.time()
execution_time = end_time - start_time
print("Execution Time               :", execution_time, "seconds")

print("\n")