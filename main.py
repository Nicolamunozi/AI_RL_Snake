import os
import random
from collections import deque

import numpy as np
import torch

from helper import plot
from model import Linear_Qnet, QTrainer, device
from snake import SnakeAI, NORTH, SOUTH, WEST, EAST

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
CHECKPOINT_FOLDER = './model'
CHECKPOINT_FILE = 'last_checkpoint.pth'
BEST_MODEL_FILE = 'model.pth'
MIN_EPSILON = 0.02
MAX_EPSILON = 0.90

print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = MAX_EPSILON
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Qnet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.games_since_record = 0

    def get_state(self, game):
        head = game.snake.head
        head_current_position = head.pos()
        head_current_orientation = head.heading()
        food = game.food

        west_point_pos = (head_current_position[0] - 20, head_current_position[1])
        east_point_pos = (head_current_position[0] + 20, head_current_position[1])
        south_point_pos = (head_current_position[0], head_current_position[1] - 20)
        north_point_pos = (head_current_position[0], head_current_position[1] + 20)

        towards_west = head_current_orientation == WEST
        towards_east = head_current_orientation == EAST
        towards_south = head_current_orientation == SOUTH
        towards_north = head_current_orientation == NORTH

        is_not_eating_food = head.distance(food) > 15

        state = [
            (towards_north and game.is_collision(*north_point_pos)) or
            (towards_east and game.is_collision(*east_point_pos)) or
            (towards_south and game.is_collision(*south_point_pos)) or
            (towards_west and game.is_collision(*west_point_pos)),

            (towards_north and game.is_collision(*east_point_pos)) or
            (towards_east and game.is_collision(*south_point_pos)) or
            (towards_south and game.is_collision(*west_point_pos)) or
            (towards_west and game.is_collision(*north_point_pos)),

            (towards_north and game.is_collision(*west_point_pos)) or
            (towards_east and game.is_collision(*north_point_pos)) or
            (towards_south and game.is_collision(*east_point_pos)) or
            (towards_west and game.is_collision(*south_point_pos)),

            towards_west,
            towards_east,
            towards_north,
            towards_south,
            food.xcor() < head.xcor() and is_not_eating_food,
            food.xcor() > head.xcor() and is_not_eating_food,
            food.ycor() > head.ycor() and is_not_eating_food,
            food.ycor() < head.ycor() and is_not_eating_food,
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_is_over):
        self.memory.append((state, action, reward, next_state, game_is_over))

    def train_long_memory(self):
        mini_sample = random.sample(self.memory, BATCH_SIZE) if len(self.memory) > BATCH_SIZE else self.memory
        states, actions, rewards, next_states, game_is_ons = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_is_ons)

    def train_short_memory(self, state, action, reward, next_state, game_is_on):
        self.trainer.train_step(state, action, reward, next_state, game_is_on)

    def update_epsilon(self, recent_scores, record):
        recent_mean = float(sum(recent_scores) / len(recent_scores)) if recent_scores else 0.0
        if self.n_games < 50:
            epsilon = max(0.25, MAX_EPSILON - self.n_games * 0.01)
        elif recent_mean >= max(8, record * 0.6):
            epsilon = max(MIN_EPSILON, 0.18 - min(0.12, recent_mean * 0.005))
        elif recent_mean >= 4:
            epsilon = max(0.05, 0.22 - recent_mean * 0.01)
        else:
            epsilon = max(0.10, 0.35 - self.n_games * 0.001)

        if self.games_since_record >= 75:
            epsilon = min(MAX_EPSILON, epsilon + 0.10)
        elif self.games_since_record >= 40:
            epsilon = min(MAX_EPSILON, epsilon + 0.05)

        self.epsilon = float(max(MIN_EPSILON, min(MAX_EPSILON, epsilon)))

    def get_action(self, state, mode='train'):
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


def get_checkpoint_path(file_name):
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
    return os.path.join(CHECKPOINT_FOLDER, file_name)


def checkpoint_exists():
    return os.path.isfile(get_checkpoint_path(CHECKPOINT_FILE))


def best_model_exists():
    return os.path.isfile(get_checkpoint_path(BEST_MODEL_FILE))


def save_checkpoint(agent, record, plot_scores, plot_mean_scores):
    checkpoint = {
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.trainer.optimizer.state_dict(),
        'n_games': agent.n_games,
        'record': record,
        'epsilon': agent.epsilon,
        'scores': plot_scores,
        'mean_scores': plot_mean_scores,
        'games_since_record': agent.games_since_record,
        'memory': list(agent.memory),
    }
    torch.save(checkpoint, get_checkpoint_path(CHECKPOINT_FILE))


def load_checkpoint(agent):
    path = get_checkpoint_path(CHECKPOINT_FILE)
    if not os.path.isfile(path):
        return None

    checkpoint = torch.load(path, map_location=device)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.model.to(device)
    agent.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.n_games = checkpoint.get('n_games', 0)
    agent.epsilon = checkpoint.get('epsilon', MAX_EPSILON)
    agent.games_since_record = checkpoint.get('games_since_record', 0)
    memory = checkpoint.get('memory', [])
    agent.memory = deque(memory, maxlen=MAX_MEMORY)

    return {
        'record': checkpoint.get('record', 0),
        'plot_scores': checkpoint.get('scores', []),
        'plot_mean_scores': checkpoint.get('mean_scores', []),
    }


def load_best_model_only(agent):
    return agent.model.load(BEST_MODEL_FILE)


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeAI()

    has_checkpoint = checkpoint_exists()
    checkpoint_preview = None
    if has_checkpoint:
        raw = torch.load(get_checkpoint_path(CHECKPOINT_FILE), map_location='cpu')
        scores = raw.get('scores', [])
        checkpoint_preview = {
            'n_games': raw.get('n_games', 0),
            'record': raw.get('record', 0),
            'mean_score': (sum(scores) / len(scores)) if scores else 0.0,
        }

    game.show_start_menu(has_checkpoint=has_checkpoint, checkpoint_info=checkpoint_preview)
    choice = game.wait_for_mode_selection()

    mode = 'TRAIN'
    loaded = False

    if choice == 'train' and has_checkpoint:
        loaded_data = load_checkpoint(agent)
        if loaded_data:
            loaded = True
            plot_scores = loaded_data['plot_scores']
            plot_mean_scores = loaded_data['plot_mean_scores']
            record = loaded_data['record']
            total_score = sum(plot_scores)
    elif choice == 'model':
        loaded = load_best_model_only(agent)
        mode = 'MODEL'
        agent.epsilon = 0.0
    elif choice == 'new':
        mode = 'TRAIN'
    else:
        if choice == 'train' and not has_checkpoint:
            mode = 'TRAIN'
        elif choice == 'model':
            loaded = load_best_model_only(agent)
            mode = 'MODEL'
            agent.epsilon = 0.0

    if plot_scores and plot_mean_scores:
        plot(plot_scores, plot_mean_scores)

    recent_scores = deque(plot_scores[-100:], maxlen=100)
    model_moves = 0
    random_moves = 0

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

        state_old = agent.get_state(game)
        final_move, used = agent.get_action(state_old, mode=mode.lower())
        game_is_over, score, reward = game.play_game(final_move, record=record)
        state_new = agent.get_state(game)

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
            total_score += score
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

            if mode == 'TRAIN' and (agent.n_games % 25 == 0 or score == record):
                save_checkpoint(agent, record, plot_scores, plot_mean_scores)

            total_moves = max(1, model_moves + random_moves)
            print(
                f'Game: {agent.n_games} | Score: {score} | Record: {record} | Mean: {mean_score:.2f} | '
                f'Epsilon: {agent.epsilon:.3f} | %Model: {model_moves/total_moves:.0%} | Mode: {mode}'
            )
            model_moves, random_moves = 0, 0


if __name__ == '__main__':
    train()
