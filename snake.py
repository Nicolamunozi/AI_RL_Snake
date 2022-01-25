from turtle import Turtle, Screen
from food import Food
from scoreboard import ScoreBoard 
import time 
import numpy as np 



INITIAL_POSITIONS = (0,0) #(x,y) coordinates.
MOVE_DISTANCE = 20
NORTH = 90
SOUTH = 270
EAST = 0
WEST = 180


class Snake():
    """This is a class for  work with the snake body for the snake game."""
    
    
    def __init__(self):
        self.snake_shards = []
        self.create_snake()
        self.head = self.snake_shards[0]
        self.frame_iteration = 0
        
        
    def create_snake(self):
        """ This functions creates the snake."""    
        for shard in range(3):
            snake = Turtle("square")
            snake.color("White")
            snake.penup()
            snake.setposition(INITIAL_POSITIONS[0]-(20*shard), INITIAL_POSITIONS[1])
            self.snake_shards.append(snake)


    def move_snake(self):
        """This function moves all the shards of the snake"""
        for shard in range(len(self.snake_shards)-1, 0, -1):
            new_x = self.snake_shards[shard-1].xcor()
            new_y = self.snake_shards[shard-1].ycor()
            self.snake_shards[shard].goto(new_x, new_y)
        self.head.forward(MOVE_DISTANCE)
        self.frame_iteration+=1
        
    def no_turn(self):
        
        self.head.right(0)
        
    def right_turn(self):

        self.head.right(90)
        
    def left_turn(self):
        
        self.head.left(90)
 
    def up(self):
        """This function set heading up."""
        if self.head.heading() !=   SOUTH:
            self.head.setheading(NORTH)
       

    def down(self):
        """This function set heading down."""
        if self.head.heading() != NORTH:
            self.head.setheading(SOUTH)


    def right(self):
        """This function set heading right."""
        if self.head.heading() != WEST:
            self.head.setheading(EAST)


    def left(self):
        """This function set heading left."""
        if self.head.heading() != EAST:
            self.head.setheading(WEST)
        

    def growth(self):
        """This function add a shard to the snake tail."""
        #Current positions:
        last_shard_x = self.snake_shards[-1].xcor()
        last_shard_y = self.snake_shards[-1].ycor()
        semi_last_shard_x = self.snake_shards[-2].xcor()
        semi_last_shard_y = self.snake_shards[-2].ycor()
        
        #Shard creation:
        snake = Turtle("square")
        snake.color("White")
        snake.penup()
        
        #New shard position:
        if last_shard_x - semi_last_shard_x > 0:   
            snake.setposition(last_shard_x + 20, last_shard_y)
        elif last_shard_x - semi_last_shard_x < 0:
            snake.setposition(last_shard_x - 20, last_shard_y)
        else: 
            if last_shard_y - semi_last_shard_y > 0:
                snake.setposition(last_shard_x, last_shard_y+20)
            elif last_shard_y - semi_last_shard_y < 0:
                snake.setposition(last_shard_x, last_shard_y-20)

        self.snake_shards.append(snake)


class SnakeAI():

    def __init__(self):

        #Screen settings
        WIDTH = 600
        HEIGHT = 480
        self.screen = Screen()
        self.screen.setup(width=WIDTH, height=HEIGHT)
        self.screen.title("Nick's Snake Game.")
        self.screen.bgcolor("Black")
        self.screen.tracer(0)

        #Objects creation

        self.snake = Snake()
        self.food = Food()
        self.score = ScoreBoard()

    def reset_game(self):

        self.screen.reset()
        self.__init__()

    def play_game(self, action=[]):

        self.game_is_over = False
        self.movement_list = [self.snake.no_turn,
                              self.snake.right_turn, self.snake.left_turn]

        if np.array_equal(action, [1, 0, 0]):
            self.movement_list[0]()  # No turn
        elif np.array_equal(action, [0, 1, 0]):
            self.movement_list[1]()  # Turn right
        else:  # [0,1,0]
            self.movement_list[2]()  # turn left
        self.screen.update()
        time.sleep(0.001)
        self.snake.move_snake()

        # Detect colision with food.
        if self.snake.head.distance(self.food) < 15:
            self.food.refresh()
            for shard in self.snake.snake_shards:
                if self.food.position() == shard.position():
                    self.food.refresh()
            self.snake.growth()
            self.score.increase_score()

        if self.is_collision() or self.snake.frame_iteration > 100 * len(self.snake.snake_shards):
            self.score.game_over()
            self.game_is_over = True

        return self.game_is_over, self.score.score, self.score.reward

    def is_collision(self, x_cord=None, y_cord=None):
        #For default values

        if x_cord == None and y_cord == None:
            head_current_position = self.snake.head.pos()
            x_cord = head_current_position[0]
            y_cord = head_current_position[1]
        # Detect colision  with wall.
        value = False
        if x_cord > 280 or x_cord < -280 or y_cord > 220 or y_cord < -220:
            value = True
        # Detect colision with tail.
        for shard in self.snake.snake_shards[1:]:
            if self.snake.head.pos() == shard.pos():
                value = True
        return value


        
