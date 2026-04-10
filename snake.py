import time
from turtle import Screen, Turtle

import numpy as np

from food import Food
from scoreboard import ScoreBoard

INITIAL_POSITIONS = (0, 0)
MOVE_DISTANCE = 20
NORTH = 90
SOUTH = 270
EAST = 0
WEST = 180
WIDTH = 600
HEIGHT = 480


class Snake:
    """Snake body and movement logic."""

    def __init__(self):
        self.snake_shards = []
        self.frame_iteration = 0
        self.create_snake()
        self.head = self.snake_shards[0]
        self.head.shape("arrow")

    def create_snake(self):
        for shard in range(3):
            snake = Turtle("square")
            snake.color("green")
            snake.penup()
            snake.speed("fastest")
            snake.setposition(INITIAL_POSITIONS[0] - (20 * shard), INITIAL_POSITIONS[1])
            self.snake_shards.append(snake)

    def move_snake(self):
        for shard in range(len(self.snake_shards) - 1, 0, -1):
            new_x = self.snake_shards[shard - 1].xcor()
            new_y = self.snake_shards[shard - 1].ycor()
            self.snake_shards[shard].goto(new_x, new_y)
        self.head.forward(MOVE_DISTANCE)
        self.frame_iteration += 1

    def no_turn(self):
        return None

    def right_turn(self):
        self.head.right(90)

    def left_turn(self):
        self.head.left(90)

    def growth(self):
        snake = Turtle("square")
        snake.color("green")
        snake.speed("fastest")
        snake.penup()
        snake.goto(self.snake_shards[-1].position())
        self.snake_shards.append(snake)

    def hide(self):
        for shard in self.snake_shards:
            shard.hideturtle()
            shard.clear()


class SnakeAI:
    def __init__(self):
        self.screen = Screen()
        self.screen.setup(width=WIDTH, height=HEIGHT)
        self.screen.title("Nick's Snake Game")
        self.screen.bgcolor("black")
        self.screen.tracer(0)
        self.reset_game(initial=True)

    def reset_game(self, initial=False):
        if not initial:
            self.snake.hide()
            self.food.hideturtle()
            self.score.clear()
            self.score.hideturtle()

        self.snake = Snake()
        self.food = Food()
        self.score = ScoreBoard()
        self.game_is_over = False
        self.is_eating = False
        self.place_food_safely()
        self.screen.update()

    def place_food_safely(self):
        self.food.refresh()
        while any(self.food.position() == shard.position() for shard in self.snake.snake_shards):
            self.food.refresh()

    def play_game(self, action=None):
        if action is None:
            action = [1, 0, 0]

        reward = -0.1
        self.game_is_over = False

        movement_list = [self.snake.no_turn, self.snake.right_turn, self.snake.left_turn]

        if np.array_equal(action, [1, 0, 0]):
            movement_list[0]()
        elif np.array_equal(action, [0, 1, 0]):
            movement_list[1]()
        else:
            movement_list[2]()

        self.snake.move_snake()

        if self.snake.head.distance(self.food) < 15:
            self.score.increase_score()
            self.snake.growth()
            self.snake.frame_iteration = 0
            self.is_eating = True
            reward = 10
            self.place_food_safely()
        else:
            self.is_eating = False

        if self.is_collision() or self.snake.frame_iteration > 100 * len(self.snake.snake_shards):
            self.score.game_over()
            self.game_is_over = True
            reward = -10

        self.screen.update()
        time.sleep(0.0001)
        return self.game_is_over, self.score.score, reward

    def is_collision(self, x_cord=None, y_cord=None):
        if x_cord is None or y_cord is None:
            x_cord, y_cord = self.snake.head.pos()

        if x_cord > 280 or x_cord < -280 or y_cord > 220 or y_cord < -220:
            return True

        for shard in self.snake.snake_shards[1:]:
            if (x_cord, y_cord) == shard.pos():
                return True
        return False
