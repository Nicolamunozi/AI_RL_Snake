import os
import random
from collections import deque

import numpy as np
import torch

from config import (
    MAX_MEMORY, BATCH_SIZE, LR, GAMMA,
    MIN_EPSILON, MAX_EPSILON,
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
    TARGET_UPDATE_FREQ,
    CHECKPOINT_FOLDER, CHECKPOINT_FILE, BEST_MODEL_FILE,
)
from helper import plot
from model import Linear_Qnet, QTrainer, device
from snake import SnakeAI, NORTH, SOUTH, WEST, EAST

print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))


# =============================================================================
# Agent
# =============================================================================

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = MAX_EPSILON          # Exploration rate (decays over time)
        self.gamma = GAMMA                  # Discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Qnet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma,
                                target_update_freq=TARGET_UPDATE_FREQ)
        self.games_since_record = 0

    def get_state(self, game):
        """
        Build an 11-element binary state vector describing the environment.

        Features (all 0 or 1):
          [0]   Danger straight ahead
          [1]   Danger to the right
          [2]   Danger to the left
          [3-6] Current direction: W, E, N, S
          [7]   Food is to the west
          [8]   Food is to the east
          [9]   Food is to the north
          [10]  Food is to the south
        """
        head = game.snake.head
        head_pos = head.pos()
        head_dir = head.heading()
        food = game.food

        # One-step lookahead positions in each cardinal direction
        west_pos  = (head_pos[0] - 20, head_pos[1])
        east_pos  = (head_pos[0] + 20, head_pos[1])
        south_pos = (head_pos[0],      head_pos[1] - 20)
        north_pos = (head_pos[0],      head_pos[1] + 20)

        towards_west  = head_dir == WEST
        towards_east  = head_dir == EAST
        towards_south = head_dir == SOUTH
        towards_north = head_dir == NORTH

        # Suppress food-direction flags when the head is on top of the food
        not_eating = head.distance(food) > 15

        state = [
            # Danger straight
            (towards_north and game.is_collision(*north_pos)) or
            (towards_east  and game.is_collision(*east_pos))  or
            (towards_south and game.is_collision(*south_pos)) or
            (towards_west  and game.is_collision(*west_pos)),

            # Danger right (clockwise 90°)
            (towards_north and game.is_collision(*east_pos))  or
            (towards_east  and game.is_collision(*south_pos)) or
            (towards_south and game.is_collision(*west_pos))  or
            (towards_west  and game.is_collision(*north_pos)),

            # Danger left (counter-clockwise 90°)
            (towards_north and game.is_collision(*west_pos))  or
            (towards_east  and game.is_collision(*north_pos)) or
            (towards_south and game.is_collision(*east_pos))  or
            (towards_west  and game.is_collision(*south_pos)),

            # Current heading
            towards_west,
            towards_east,
            towards_north,
            towards_south,

            # Food direction relative to head
            food.xcor() < head.xcor() and not_eating,   # food west
            food.xcor() > head.xcor() and not_eating,   # food east
            food.ycor() > head.ycor() and not_eating,   # food north
            food.ycor() < head.ycor() and not_eating,   # food south
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_is_over):
        """Store one transition in the circular replay buffer."""
        self.memory.append((state, action, reward, next_state, game_is_over))

    def train_long_memory(self):
        """Sample a random mini-batch from memory and train on it (off-policy)."""
        mini_sample = (
            random.sample(self.memory, BATCH_SIZE)
            if len(self.memory) > BATCH_SIZE
            else list(self.memory)
        )
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, game_is_on):
        """Train on the single most recent transition (online learning)."""
        self.trainer.train_step(state, action, reward, next_state, game_is_on)

    def update_epsilon(self, recent_scores, record):
        """
        Adaptive ε schedule:
          - First 50 games: linear decay from MAX_EPSILON → 0.25  (exploration phase)
          - After 50 games: performance-based — lower ε when the agent is doing well,
            higher ε when stuck.
          - Stagnation boost: if no new record for 40+ games, add extra exploration
            to help escape local optima.
        """
        recent_mean = float(sum(recent_scores) / len(recent_scores)) if recent_scores else 0.0

        if self.n_games < 50:
            epsilon = max(0.25, MAX_EPSILON - self.n_games * 0.013)
        elif recent_mean >= max(8, record * 0.6):
            # Performing well — exploit more
            epsilon = max(MIN_EPSILON, 0.18 - min(0.12, recent_mean * 0.005))
        elif recent_mean >= 4:
            # Mediocre — moderate exploration
            epsilon = max(0.05, 0.22 - recent_mean * 0.01)
        else:
            # Poor — explore more
            epsilon = max(0.10, 0.35 - self.n_games * 0.001)

        # Stagnation: boost exploration if no record improvement for a while
        if self.games_since_record >= 75:
            epsilon = min(MAX_EPSILON, epsilon + 0.10)
        elif self.games_since_record >= 40:
            epsilon = min(MAX_EPSILON, epsilon + 0.05)

        self.epsilon = float(max(MIN_EPSILON, min(MAX_EPSILON, epsilon)))

    def get_action(self, state, mode='train'):
        """
        ε-greedy action selection.
        Returns a one-hot action vector [straight, right, left] and a label
        indicating whether the move was random or model-driven.
        """
        final_move = [0, 0, 0]
        used = 'model'

        if mode == 'train' and random.random() < self.epsilon:
            move = random.randint(0, 2)
            used = 'random'
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move, used


# =============================================================================
# Checkpoint helpers
# =============================================================================

def get_checkpoint_path(file_name):
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
    return os.path.join(CHECKPOINT_FOLDER, file_name)


def checkpoint_exists():
    return os.path.isfile(get_checkpoint_path(CHECKPOINT_FILE))


def best_model_exists():
    return os.path.isfile(get_checkpoint_path(BEST_MODEL_FILE))


def save_checkpoint(agent, record, plot_scores, plot_mean_scores):
    """Persist the full training state so training can resume exactly where it left off."""
    checkpoint = {
        'model_state_dict':     agent.model.state_dict(),
        'optimizer_state_dict': agent.trainer.optimizer.state_dict(),
        'n_games':              agent.n_games,
        'record':               record,
        'epsilon':              agent.epsilon,
        'scores':               plot_scores,
        'mean_scores':          plot_mean_scores,
        'games_since_record':   agent.games_since_record,
        'memory':               list(agent.memory),
    }
    torch.save(checkpoint, get_checkpoint_path(CHECKPOINT_FILE))


def load_checkpoint(agent):
    """
    Restore all training state from disk.
    Returns a dict with record, plot_scores, plot_mean_scores, or None on failure.
    """
    path = get_checkpoint_path(CHECKPOINT_FILE)
    if not os.path.isfile(path):
        return None

    checkpoint = torch.load(path, map_location=device)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.model.to(device)
    agent.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.n_games           = checkpoint.get('n_games', 0)
    agent.epsilon           = checkpoint.get('epsilon', MAX_EPSILON)
    agent.games_since_record = checkpoint.get('games_since_record', 0)
    agent.memory = deque(checkpoint.get('memory', []), maxlen=MAX_MEMORY)

    return {
        'record':          checkpoint.get('record', 0),
        'plot_scores':     checkpoint.get('scores', []),
        'plot_mean_scores': checkpoint.get('mean_scores', []),
    }


def load_best_model_only(agent):
    return agent.model.load(BEST_MODEL_FILE)


# =============================================================================
# Training loop
# =============================================================================

def train():
    plot_scores      = []
    plot_mean_scores = []
    total_score      = 0
    record           = 0
    agent            = Agent()
    game             = SnakeAI()

    # --- Start-up menu -------------------------------------------------------
    has_checkpoint = checkpoint_exists()
    checkpoint_preview = None
    if has_checkpoint:
        raw    = torch.load(get_checkpoint_path(CHECKPOINT_FILE), map_location='cpu')
        scores = raw.get('scores', [])
        checkpoint_preview = {
            'n_games':    raw.get('n_games', 0),
            'record':     raw.get('record', 0),
            'mean_score': (sum(scores) / len(scores)) if scores else 0.0,
        }

    game.show_start_menu(has_checkpoint=has_checkpoint,
                         checkpoint_info=checkpoint_preview)
    choice = game.wait_for_mode_selection()

    mode   = 'TRAIN'
    loaded = False

    if choice == 'train' and has_checkpoint:
        loaded_data = load_checkpoint(agent)
        if loaded_data:
            loaded           = True
            plot_scores      = loaded_data['plot_scores']
            plot_mean_scores = loaded_data['plot_mean_scores']
            record           = loaded_data['record']
            total_score      = sum(plot_scores)
    elif choice == 'model':
        loaded = load_best_model_only(agent)
        mode   = 'MODEL'
        agent.epsilon = 0.0
    # 'new' or 'train' with no checkpoint → start fresh (defaults already set)

    if plot_scores and plot_mean_scores:
        plot(plot_scores, plot_mean_scores)

    recent_scores = deque(plot_scores[-100:], maxlen=100)
    model_moves = 0
    random_moves = 0

    # --- Main game loop ------------------------------------------------------
    while True:
        if mode == 'TRAIN':
            agent.update_epsilon(recent_scores, record)
        else:
            agent.epsilon = 0.0

        current_mean = plot_mean_scores[-1] if plot_mean_scores else 0.0
        game.score.update_scoreboard(record=record)
        game.score.update_status(
            n_games=agent.n_games,
            epsilon=agent.epsilon,
            mode=mode,
            mean_score=current_mean,
            loaded=loaded,
        )

        state_old              = agent.get_state(game)
        final_move, used       = agent.get_action(state_old, mode=mode.lower())
        game_is_over, score, reward = game.play_game(final_move, record=record)
        state_new              = agent.get_state(game)

        if used == 'random':
            random_moves += 1
        else:
            model_moves += 1

        if mode == 'TRAIN':
            agent.train_short_memory(state_old, final_move, reward, state_new, game_is_over)
            agent.remember(state_old, final_move, reward, state_new, game_is_over)

        if game_is_over:
            game.reset_game()
            agent.n_games += 1
            total_score   += score
            recent_scores.append(score)

            if mode == 'TRAIN':
                agent.train_long_memory()

            if score > record:
                record = score
                agent.games_since_record = 0
                agent.model.save(BEST_MODEL_FILE)
            else:
                agent.games_since_record += 1

            plot_scores.append(score)
            mean_score = total_score / max(agent.n_games, 1)
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Save checkpoint every 25 games or on a new record
            if mode == 'TRAIN' and (agent.n_games % 25 == 0 or score == record):
                save_checkpoint(agent, record, plot_scores, plot_mean_scores)

            total_moves = max(1, model_moves + random_moves)
            print(
                f'Game: {agent.n_games} | Score: {score} | Record: {record} | '
                f'Mean: {mean_score:.2f} | Epsilon: {agent.epsilon:.3f} | '
                f'%Model: {model_moves / total_moves:.0%} | Mode: {mode}'
            )
            model_moves, random_moves = 0, 0


if __name__ == '__main__':
    train()
