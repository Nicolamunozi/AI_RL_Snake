# =============================================================================
# Snake AI – Reinforcement Learning Configuration
# =============================================================================
# All tunable hyperparameters are centralised here.
# Edit a value here and it propagates to the entire training pipeline.

# --------------- Replay Memory -----------------------------------------------
MAX_MEMORY = 100_000    # Maximum transitions stored in the replay buffer
BATCH_SIZE = 1000       # Transitions sampled per long-memory training step

# --------------- Optimiser ---------------------------------------------------
LR = 0.001              # Adam learning rate

# --------------- Q-Learning --------------------------------------------------
GAMMA = 0.95            # Discount factor (higher = more far-sighted planning)
                        # Raised from 0.9 → 0.95 so the agent values future
                        # rewards more, improving food-chasing behaviour.

# --------------- Exploration (ε-greedy) --------------------------------------
MIN_EPSILON = 0.02      # Floor: agent always keeps a tiny random chance
MAX_EPSILON = 0.90      # Ceiling: fully random at the start of training

# --------------- Network Architecture ----------------------------------------
INPUT_SIZE  = 11        # Binary state features (see get_state() in main.py)
HIDDEN_SIZE = 256       # Neurons in the single hidden layer
OUTPUT_SIZE = 3         # Discrete actions: [straight, turn-right, turn-left]

# --------------- Target Network ----------------------------------------------
TARGET_UPDATE_FREQ = 200  # Copy online → target network every N train steps
                          # Stabilises Q-targets, reducing training oscillation.

# --------------- Reward Shaping ----------------------------------------------
REWARD_FOOD    =  10    # Reward for eating food
REWARD_DEATH   = -10    # Penalty for wall/body collision or timeout
REWARD_TOWARD  =   1    # Shaping bonus: moved closer to food this step
REWARD_AWAY    =  -1    # Shaping penalty: moved farther from food this step
# Note: distance shaping is only applied on steps where no food is eaten
# and the game has not ended, so it never interferes with REWARD_FOOD/DEATH.

# --------------- Checkpoint Paths --------------------------------------------
CHECKPOINT_FOLDER = './model'
CHECKPOINT_FILE   = 'last_checkpoint.pth'
BEST_MODEL_FILE   = 'model.pth'
