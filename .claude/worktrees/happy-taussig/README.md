# Snake AI — Deep Q-Network Reinforcement Learning

A Snake game agent trained from scratch using **Deep Q-Learning (DQN)** with experience replay, a target network, and distance-based reward shaping. The agent learns purely by playing — no hard-coded rules, no hand-crafted strategy.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Project Structure](#project-structure)
3. [State Representation](#state-representation)
4. [Actions](#actions)
5. [Reward Function](#reward-function)
6. [Neural Network Architecture](#neural-network-architecture)
7. [Key DQN Improvements](#key-dqn-improvements)
8. [Hyperparameters](#hyperparameters)
9. [Epsilon Schedule](#epsilon-schedule)
10. [Installation & Running](#installation--running)
11. [Training Modes](#training-modes)
12. [Checkpointing](#checkpointing)
13. [Optimization History](#optimization-history)

---

## How It Works

The agent uses **Deep Q-Learning (DQN)**, a value-based reinforcement learning algorithm. At every game step:

1. The environment returns an **11-feature binary state** describing dangers, direction, and food location.
2. The agent feeds this state through a neural network that outputs a **Q-value for each of 3 actions**.
3. Using an **ε-greedy policy**, the agent picks the action with the highest Q-value (or a random action with probability ε).
4. The resulting `(state, action, reward, next_state, done)` transition is stored in a **replay buffer**.
5. At the end of each episode, a random **mini-batch** is sampled and used to update the network via the **Bellman equation**:

```
Q*(s, a) = r + γ · max_a' Q_target(s', a')    [if not terminal]
Q*(s, a) = r                                    [if terminal]
```

The network gradually learns which actions lead to higher long-term rewards.

---

## Project Structure

```
AI_RL_Snake/
├── config.py       ← ALL hyperparameters (start here to tune)
├── main.py         ← Agent class, training loop, checkpoint I/O
├── model.py        ← Neural network (Linear_Qnet) + DQN trainer (QTrainer)
├── snake.py        ← Game environment: Snake, SnakeAI, collision, rewards
├── scoreboard.py   ← HUD display (score, stats, menu)
├── food.py         ← Food spawning logic
├── helper.py       ← Matplotlib training-progress plot
└── model/
    ├── model.pth              ← Best model weights (used in MODEL mode)
    └── last_checkpoint.pth    ← Full training state (resumes training)
```

---

## State Representation

The agent receives **11 binary (0/1) features** at every step:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Danger straight | Will the next step in the current direction cause a collision? |
| 1 | Danger right | Collision if turning 90° clockwise? |
| 2 | Danger left | Collision if turning 90° counter-clockwise? |
| 3 | Moving west | Current heading is west |
| 4 | Moving east | Current heading is east |
| 5 | Moving north | Current heading is north |
| 6 | Moving south | Current heading is south |
| 7 | Food west | Food is to the left of the head |
| 8 | Food east | Food is to the right of the head |
| 9 | Food north | Food is above the head |
| 10 | Food south | Food is below the head |

Features 3–6 form a one-hot encoding of direction (exactly one is 1).  
Features 7–10 are suppressed while the head is on top of the food (eating transition).

---

## Actions

Three discrete actions, encoded as one-hot vectors:

| Vector | Meaning |
|--------|---------|
| `[1, 0, 0]` | Continue straight |
| `[0, 1, 0]` | Turn right (clockwise 90°) |
| `[0, 0, 1]` | Turn left (counter-clockwise 90°) |

The action space is relative to the current heading, not absolute — this keeps the problem symmetric and halves the effective state/action complexity.

---

## Reward Function

| Event | Reward | Where set |
|-------|--------|-----------|
| Eat food | **+10** | `scoreboard.py → increase_score()` |
| Collision / timeout | **−10** | `scoreboard.py → game_over()` |
| Step toward food | **+1** | `snake.py → play_game()` (distance shaping) |
| Step away from food | **−1** | `snake.py → play_game()` (distance shaping) |

**Distance shaping** is the key addition: on every neutral step (no food eaten, no death), the reward is ±1 depending on whether the snake moved closer to or farther from the food. This gives the network a gradient signal on *every* step instead of only at food/death events, dramatically speeding up early learning.

The shaping rewards never conflict with food/death rewards because `game_over()` overwrites the reward to −10 and `increase_score()` overwrites it to +10 after the step reward is applied.

---

## Neural Network Architecture

```
Input (11)  →  Linear(11→256)  →  ReLU  →  Linear(256→3)  →  Q-values (3)
```

- **Input:** 11 binary features
- **Hidden:** 256 neurons, ReLU activation
- **Output:** 3 raw Q-values (no activation — these are value estimates, not probabilities)

The shallow architecture is intentional. The state space is small (2^11 = 2048 possible states) and extra depth would only slow training without improving the policy.

---

## Key DQN Improvements

### 1. Target Network
A **frozen copy** of the online network (`target_model`) is used to compute Bellman targets. Its weights are hard-copied from the online network every `TARGET_UPDATE_FREQ` gradient steps.

Without this, the network chases a moving target — every weight update changes both the prediction *and* the target, which destabilises training. The target network breaks this feedback loop.

### 2. Huber Loss (SmoothL1)
Replaces MSE. Behaves like MSE for small errors (quadratic, smooth gradient) and like MAE for large errors (linear, bounded gradient). This prevents large reward outliers (e.g., death at −10) from causing disproportionately large gradient updates.

### 3. Vectorised Q-Target Computation
The original code looped over each sample in the mini-batch with a Python `for` loop. The new code computes Bellman targets for the entire batch in one tensor operation:

```python
max_next_q = target_model(next_states).max(dim=1).values
q_targets  = reward + gamma * max_next_q * (1 - done)
```

This is ~10–50× faster on CPU and even more on GPU.

### 4. Gradient Clipping
Clips the gradient L2-norm to 1.0 before each optimizer step:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Prevents rare but catastrophic parameter updates when the loss surface is steep.

---

## Hyperparameters

All values live in **`config.py`**. Change them there — no other file needs editing.

| Parameter | Value | Effect |
|-----------|-------|--------|
| `GAMMA` | **0.95** | Discount factor. Higher = agent values future rewards more. Raised from 0.9 to make the snake plan ahead (navigate toward food across many steps). |
| `LR` | 0.001 | Adam learning rate. Governs how fast weights change per gradient step. |
| `BATCH_SIZE` | 1000 | Transitions sampled per long-memory training step. Larger = more stable gradients, slower per step. |
| `MAX_MEMORY` | 100,000 | Replay buffer capacity. Older transitions are discarded (FIFO). |
| `MIN_EPSILON` | 0.02 | Minimum exploration rate — agent always keeps a 2% chance of random action. |
| `MAX_EPSILON` | 0.90 | Starting exploration rate — nearly fully random at game 0. |
| `HIDDEN_SIZE` | 256 | Neurons in the single hidden layer. |
| `TARGET_UPDATE_FREQ` | 200 | Copy online → target network every N training steps. |
| `REWARD_FOOD` | +10 | Reward magnitude for eating food. |
| `REWARD_DEATH` | −10 | Penalty magnitude for dying. |
| `REWARD_TOWARD` | +1 | Distance shaping bonus per step. |
| `REWARD_AWAY` | −1 | Distance shaping penalty per step. |

---

## Epsilon Schedule

ε controls the exploration/exploitation trade-off (probability of taking a random action):

```
Games 0–50:   ε decays linearly from 0.90 → 0.25   (exploration phase)
After game 50:
  recent mean ≥ max(8, record×0.6) → ε ≈ 0.02–0.18  (exploit, performing well)
  recent mean ≥ 4                  → ε ≈ 0.05–0.18  (moderate)
  recent mean < 4                  → ε ≈ 0.10–0.35  (explore more)

Stagnation boost:
  No record for 40+ games  → ε += 0.05  (escape local optimum)
  No record for 75+ games  → ε += 0.10
```

This adaptive schedule avoids two failure modes: premature exploitation (converging to a bad policy before enough exploration) and excessive exploration (never committing to a learned policy).

---

## Installation & Running

### Requirements

```
Python 3.10+
torch
numpy
matplotlib
```

Install dependencies:
```bash
pip install torch numpy matplotlib
```

*(On Windows, `turtle` is part of the standard library — no extra install needed.)*

### Run

```bash
python main.py
```

A window will open showing the Snake game. Press a key to choose a mode:

| Key | Mode |
|-----|------|
| **T** | Train (continue from last checkpoint, or start fresh if none) |
| **M** | Model-only (watch the trained agent play, no learning) |
| **N** | New training (discard checkpoint, start from scratch) |

---

## Training Modes

### TRAIN mode
- ε-greedy policy (exploration + exploitation)
- Short-memory training after every step
- Long-memory (replay) training after every episode
- Checkpoint saved every 25 games or on a new record
- Live score plot shown in a separate window

### MODEL mode
- ε = 0 (pure exploitation — always picks the best known action)
- No training, no checkpoint writes
- Uses `model/model.pth` (best weights ever saved)

---

## Checkpointing

Two files are saved to `./model/`:

| File | Content | When saved |
|------|---------|-----------|
| `model.pth` | Best model weights only | On every new record |
| `last_checkpoint.pth` | Full state: weights, optimizer, replay buffer, score history, epsilon | Every 25 games or on a new record |

The full checkpoint lets training resume from the exact same point — same replay memory, same optimizer momentum, same score history — as if it never stopped.

---

## Optimization History

### Baseline (original code)
- Single hidden layer, MSE loss, no target network
- Q-targets computed in a Python loop per sample
- Gamma = 0.9, no distance reward shaping
- Step reward = 0 (no gradient signal between food events)

### Optimizations applied
| Change | Rationale |
|--------|-----------|
| **Target network** (every 200 steps) | Breaks the moving-target feedback loop; reduces Q-value oscillation |
| **Huber loss** (SmoothL1) | More robust to outlier rewards than MSE |
| **Vectorised Bellman targets** | 10–50× faster batch training; cleaner code |
| **Gradient clipping** (max norm 1.0) | Prevents catastrophic weight updates |
| **Gamma 0.9 → 0.95** | Agent plans farther ahead; reaches food along longer paths |
| **Distance reward shaping** (±1/step) | Dense gradient signal on every step vs. sparse food/death only |
| **Centralised config.py** | All hyperparameters in one place; no hunting through multiple files |
| **Adaptive epsilon decay** (0.013/game) | Slightly faster early exploration collapse vs. original 0.01/game |
| **Robust reset_game()** | Cleans up turtle objects properly instead of `screen.reset()` |
| **Independent scoreboard writers** | Header, status bar, and menu cleared independently |
| **Food body-overlap check** | Food never spawns on top of the snake after eating |
| **Collision lookahead fix** | Excludes tail segment from lookahead checks (tail moves away next step) |
