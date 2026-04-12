import time

import numpy as np
from turtle import Turtle, Screen

from food import Food
from scoreboard import ScoreBoard
from config import REWARD_TOWARD, REWARD_AWAY

INITIAL_POSITIONS = (0, 0)
MOVE_DISTANCE     = 20
NORTH = 90
SOUTH = 270
EAST  = 0
WEST  = 180


class Snake:
    def __init__(self):
        self.snake_shards = []
        self.create_snake()
        self.head = self.snake_shards[0]
        self.head.shape('arrow')
        self.frame_iteration = 0

    def create_snake(self):
        for shard in range(3):
            seg = Turtle('square')
            seg.color('green')
            seg.penup()
            seg.speed('fastest')
            seg.setposition(INITIAL_POSITIONS[0] - 20 * shard, INITIAL_POSITIONS[1])
            self.snake_shards.append(seg)

    def move_snake(self):
        # Shift every segment to the position of the one ahead of it
        for idx in range(len(self.snake_shards) - 1, 0, -1):
            new_x = self.snake_shards[idx - 1].xcor()
            new_y = self.snake_shards[idx - 1].ycor()
            self.snake_shards[idx].goto(new_x, new_y)
        self.head.forward(MOVE_DISTANCE)
        self.frame_iteration += 1

    def no_turn(self):
        self.head.right(0)

    def right_turn(self):
        self.head.right(90)

    def left_turn(self):
        self.head.left(90)

    def growth(self):
        """Append a new segment behind the tail."""
        last_x  = self.snake_shards[-1].xcor()
        last_y  = self.snake_shards[-1].ycor()
        prev_x  = self.snake_shards[-2].xcor()
        prev_y  = self.snake_shards[-2].ycor()

        seg = Turtle('square')
        seg.color('green')
        seg.speed('fastest')
        seg.penup()

        if last_x - prev_x > 0:
            seg.setposition(last_x + 20, last_y)
        elif last_x - prev_x < 0:
            seg.setposition(last_x - 20, last_y)
        else:
            if last_y - prev_y > 0:
                seg.setposition(last_x, last_y + 20)
            elif last_y - prev_y < 0:
                seg.setposition(last_x, last_y - 20)

        self.snake_shards.append(seg)


class SnakeAI:
    def __init__(self):
        self.width  = 600
        self.height = 480
        self.screen = Screen()
        self.screen.setup(width=self.width, height=self.height)
        self.screen.title("Nick's Snake Game")
        self.screen.bgcolor('black')
        self.screen.tracer(0)

        self.snake        = Snake()
        self.food         = Food()
        self.score        = ScoreBoard()
        self.game_is_over = False
        self.is_eating    = False
        self.choice       = None

    def reset_game(self):
        """Tear down all turtle objects and reinitialise from scratch."""
        for shard in self.snake.snake_shards:
            shard.hideturtle()
            shard.goto(1000, 1000)
        self.food.hideturtle()
        self.score.score  = 0
        self.score.reward = 0
        self.screen.clearscreen()
        self.__init__()

    def refresh_food_if_needed(self):
        """Move food to a new position if the snake just ate, avoiding body overlap."""
        if self.is_eating:
            while True:
                self.food.refresh()
                if all(self.food.position() != shard.position()
                       for shard in self.snake.snake_shards):
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
        """Block until the user presses T, M, or N; return the choice string."""
        self.choice = None

        def set_train(): self.choice = 'train'
        def set_model(): self.choice = 'model'
        def set_new():   self.choice = 'new'

        self.screen.listen()
        self.screen.onkey(set_train, 't')
        self.screen.onkey(set_train, 'T')
        self.screen.onkey(set_model, 'm')
        self.screen.onkey(set_model, 'M')
        self.screen.onkey(set_new,   'n')
        self.screen.onkey(set_new,   'N')

        while self.choice is None:
            self.screen.update()
            time.sleep(0.05)

        self.score.clear_menu()
        return self.choice

    def play_game(self, action=None, record=0):
        """
        Execute one game step.

        Reward structure:
          - Eating food:       REWARD_FOOD  (set by ScoreBoard.increase_score)
          - Collision/timeout: REWARD_DEATH (set by ScoreBoard.game_over)
          - Moving toward food: REWARD_TOWARD  (distance shaping, applied here)
          - Moving away:        REWARD_AWAY    (distance shaping, applied here)

        Distance shaping is only applied on neutral steps (no food eaten, no death)
        so it never interferes with the terminal rewards.
        """
        action = action or [1, 0, 0]
        self.game_is_over = False
        self.score.set_step_reward()
        self.refresh_food_if_needed()

        # Record distance to food before the move (used for reward shaping below)
        dist_before = self.snake.head.distance(self.food)

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
            # Distance-based shaping: reward the agent for making progress toward food
            dist_after = self.snake.head.distance(self.food)
            self.score.reward += REWARD_TOWARD if dist_after < dist_before else REWARD_AWAY

        if self.is_collision() or self.snake.frame_iteration > 100 * len(self.snake.snake_shards):
            self.score.game_over()
            self.game_is_over = True

        self.screen.update()
        time.sleep(0.0001)
        return self.game_is_over, self.score.score, self.score.reward

    def is_collision(self, x_cord=None, y_cord=None):
        """
        Check whether a given coordinate (or the head's current position)
        is a wall or body collision.

        When called with explicit coordinates (lookahead check from get_state),
        the last body segment is excluded because the tail moves away on the
        next step — checking it would produce false positives.
        """
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
