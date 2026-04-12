from turtle import Turtle, Screen
from food import Food
from scoreboard import ScoreBoard
import time
import numpy as np

INITIAL_POSITIONS = (0, 0)
MOVE_DISTANCE = 20
NORTH = 90
SOUTH = 270
EAST = 0
WEST = 180


class Snake:
    def __init__(self):
        self.snake_shards = []
        self.create_snake()
        self.head = self.snake_shards[0]
        self.head.shape('arrow')
        self.frame_iteration = 0

    def create_snake(self):
        for shard in range(3):
            snake = Turtle('square')
            snake.color('green')
            snake.penup()
            snake.speed('fastest')
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
        self.head.right(0)

    def right_turn(self):
        self.head.right(90)

    def left_turn(self):
        self.head.left(90)

    def growth(self):
        last_shard_x = self.snake_shards[-1].xcor()
        last_shard_y = self.snake_shards[-1].ycor()
        semi_last_shard_x = self.snake_shards[-2].xcor()
        semi_last_shard_y = self.snake_shards[-2].ycor()

        snake = Turtle('square')
        snake.color('green')
        snake.speed('fastest')
        snake.penup()

        if last_shard_x - semi_last_shard_x > 0:
            snake.setposition(last_shard_x + 20, last_shard_y)
        elif last_shard_x - semi_last_shard_x < 0:
            snake.setposition(last_shard_x - 20, last_shard_y)
        else:
            if last_shard_y - semi_last_shard_y > 0:
                snake.setposition(last_shard_x, last_shard_y + 20)
            elif last_shard_y - semi_last_shard_y < 0:
                snake.setposition(last_shard_x, last_shard_y - 20)

        self.snake_shards.append(snake)


class SnakeAI:
    def __init__(self):
        self.width = 600
        self.height = 480
        self.screen = Screen()
        self.screen.setup(width=self.width, height=self.height)
        self.screen.title("Nick's Snake Game")
        self.screen.bgcolor('black')
        self.screen.tracer(0)

        self.snake = Snake()
        self.food = Food()
        self.score = ScoreBoard()
        self.game_is_over = False
        self.is_eating = False
        self.choice = None

    def reset_game(self):
        for shard in self.snake.snake_shards:
            shard.hideturtle()
            shard.goto(1000, 1000)
        self.food.hideturtle()
        self.score.score = 0
        self.score.reward = 0
        self.screen.clearscreen()
        self.__init__()

    def refresh_food_if_needed(self):
        if self.is_eating:
            while True:
                self.food.refresh()
                if all(self.food.position() != shard.position() for shard in self.snake.snake_shards):
                    break

    def show_start_menu(self, has_checkpoint=False, checkpoint_info=None):
        info = checkpoint_info or {}
        if has_checkpoint:
            lines = [
                'Checkpoint detected',
                f"Games: {info.get('n_games', 0)} | Record: {info.get('record', 0)} | Mean: {info.get('mean_score', 0.0):.2f}",
                'Press T = continue training',
                'Press M = use model only',
                'Press N = new training',
            ]
        else:
            lines = [
                'No checkpoint found',
                'Press T = start training from scratch',
                'Press M = use saved best model only (if available)',
            ]
        self.score.show_menu(lines)
        self.screen.update()

    def wait_for_mode_selection(self):
        self.choice = None

        def set_train():
            self.choice = 'train'

        def set_model():
            self.choice = 'model'

        def set_new():
            self.choice = 'new'

        self.screen.listen()
        self.screen.onkey(set_train, 't')
        self.screen.onkey(set_train, 'T')
        self.screen.onkey(set_model, 'm')
        self.screen.onkey(set_model, 'M')
        self.screen.onkey(set_new, 'n')
        self.screen.onkey(set_new, 'N')

        while self.choice is None:
            self.screen.update()
            time.sleep(0.05)

        self.score.clear_menu()
        return self.choice

    def play_game(self, action=None, record=0):
        action = action or [1, 0, 0]
        self.game_is_over = False
        self.score.set_step_reward()
        self.refresh_food_if_needed()

        movement_list = [self.snake.no_turn, self.snake.right_turn, self.snake.left_turn]

        if np.array_equal(action, [1, 0, 0]):
            movement_list[0]()
        elif np.array_equal(action, [0, 1, 0]):
            movement_list[1]()
        else:
            movement_list[2]()

        self.snake.move_snake()

        if self.snake.head.distance(self.food) < 15:
            self.score.increase_score(record=record)
            self.snake.growth()
            self.snake.frame_iteration = 0
            self.is_eating = True
        else:
            self.is_eating = False

        if self.is_collision() or self.snake.frame_iteration > 100 * len(self.snake.snake_shards):
            self.score.game_over()
            self.game_is_over = True

        self.screen.update()
        time.sleep(0.0001)
        return self.game_is_over, self.score.score, self.score.reward

    def is_collision(self, x_cord=None, y_cord=None):
        if x_cord is None and y_cord is None:
            x_cord = self.snake.head.xcor()
            y_cord = self.snake.head.ycor()
            body_to_check = self.snake.snake_shards[1:]
        else:
            body_to_check = self.snake.snake_shards[:-1]

        if x_cord > 280 or x_cord < -280 or y_cord > 220 or y_cord < -220:
            return True

        for shard in body_to_check:
            if (x_cord, y_cord) == shard.pos():
                return True
        return False
