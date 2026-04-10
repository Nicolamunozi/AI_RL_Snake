import random
from collections import deque

import numpy as np
import torch

from helper import plot
from model import DEVICE, Linear_Qnet, QTrainer
from snake import EAST, NORTH, SOUTH, WEST, SnakeAI

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
FOREPSILON = 80

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Qnet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeAI) -> np.ndarray:
        head = game.snake.head
        head_x, head_y = head.pos()
        heading = head.heading()
        food = game.food

        west_point = (head_x - 20, head_y)
        east_point = (head_x + 20, head_y)
        south_point = (head_x, head_y - 20)
        north_point = (head_x, head_y + 20)

        towards_west = heading == WEST
        towards_east = heading == EAST
        towards_south = heading == SOUTH
        towards_north = heading == NORTH

        state = [
            # Danger straight
            (towards_north and game.is_collision(*north_point))
            or (towards_east and game.is_collision(*east_point))
            or (towards_south and game.is_collision(*south_point))
            or (towards_west and game.is_collision(*west_point)),
            # Danger right
            (towards_north and game.is_collision(*east_point))
            or (towards_east and game.is_collision(*south_point))
            or (towards_south and game.is_collision(*west_point))
            or (towards_west and game.is_collision(*north_point)),
            # Danger left
            (towards_north and game.is_collision(*west_point))
            or (towards_east and game.is_collision(*north_point))
            or (towards_south and game.is_collision(*east_point))
            or (towards_west and game.is_collision(*south_point)),
            # Current direction
            towards_west,
            towards_east,
            towards_north,
            towards_south,
            # Food location relative to head
            food.xcor() < head.xcor(),
            food.xcor() > head.xcor(),
            food.ycor() > head.ycor(),
            food.ycor() < head.ycor(),
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_is_over):
        self.memory.append((state, action, reward, next_state, game_is_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)

        if not mini_sample:
            return

        states, actions, rewards, next_states, game_is_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_is_overs)

    def train_short_memory(self, state, action, reward, next_state, game_is_over):
        self.trainer.train_step(state, action, reward, next_state, game_is_over)

    def get_action(self, state):
        self.epsilon = max(0, FOREPSILON - self.n_games)
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            source = "random"
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=DEVICE)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            source = "model"

        final_move[move] = 1
        return final_move, source


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    agent.model.load()
    game = SnakeAI()

    tm, tr = 0, 0

    while True:
        state_old = agent.get_state(game)
        final_move, source = agent.get_action(state_old)
        game_is_over, score, reward = game.play_game(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, game_is_over)
        agent.remember(state_old, final_move, reward, state_new, game_is_over)

        if source == "random":
            tr += 1
        else:
            tm += 1

        tt = tr + tm

        if game_is_over:
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"Game: {agent.n_games}. Score: {score}. Record: {record}. %M: {tm / tt:.0%}.")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            tm, tr = 0, 0
            game.reset_game()


if __name__ == "__main__":
    train()
