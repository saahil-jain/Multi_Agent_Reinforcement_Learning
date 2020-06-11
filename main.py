import os

print("\n2 AGENTS :\n")

print("Using Same Q-tables :")
os.system("python3 trainer_two_common.py")

print("\nUsing Different Q-tables :")
os.system("python3 trainer_two.py")

print("\n3 AGENTS :\n")

print("Using Same Q-tables :")
os.system("python3 trainer_three_common.py")

print("\nUsing Different Q-tables :")
os.system("python3 trainer_three.py")

print("\n")

# Q[S][A] = ((1- LEARNING_RATE) * Q[S][A]) + (LEARNING_RATE * (Current_Reward + (DISCOUNT * max(Q[Perform_action(A)]))))