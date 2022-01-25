
from snake import Snake, SnakeAI, NORTH, SOUTH, WEST, EAST
from food import Food
from scoreboard import ScoreBoard
import random 
import numpy as np 
import torch
from collections import deque 
from model import Linear_Qnet, QTrainer
from helper import plot


#constants:
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.5
FOREPSILON = 80

        
class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomess
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #pop left 
        self.model = Linear_Qnet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    
    def get_state(self, game): #This is gonna be usefull fpr the IA to understand the game.
        """
        TODO: 
        1) Create points around the head. check.
        2) Check if there is danger of collision with those points. check 
        3) Check the direction of movement. check 
        4) Check where is the food. (to the left, right,  top or bottom) check
        5) Return a boolean matrix (check).
        
        """
        head = game.snake.head
        head_current_position = game.snake.head.pos()
        head_current_orientation =  game.snake.head.heading()
        food = game.food
        food_x = food.pos()[0]
        food_y = food.pos()[1]
         
        # Points  positions around the head. (tuples) 
        west_point_pos = (head_current_position[0] - 20, head_current_position[1])
        east_point_pos = (head_current_position[0] + 20, head_current_position[1])
        south_point_pos = (head_current_position[0], head_current_position[1] - 20)
        north_point_pos = (head_current_position[0], head_current_position[1] + 20) 
        
        # Boleans of direction.
        towards_west = head_current_orientation == WEST
        towards_east = head_current_orientation == EAST
        towards_south = head_current_orientation == SOUTH
        towards_north = head_current_orientation == NORTH
        
        #Now is time to use an array of state.
        state = [
            #Danger Front: 
            towards_north and game.is_collision(north_point_pos[0], north_point_pos[1]) or 
            towards_east  and game.is_collision(east_point_pos[0], east_point_pos[1])  or 
            towards_south and game.is_collision(south_point_pos[0], south_point_pos[1]) or
            towards_west  and game.is_collision(west_point_pos[0], south_point_pos[1]),
            
            #Danger Right:
            towards_north and game.is_collision(east_point_pos[0], east_point_pos[1]) or 
            towards_east  and game.is_collision(south_point_pos[0], south_point_pos[1]) or 
            towards_south and game.is_collision(west_point_pos[0], west_point_pos[1]) or 
            towards_west  and game.is_collision(north_point_pos[0],north_point_pos[1]),
             
            #Danger Left: 
            towards_north and game.is_collision(west_point_pos[0], west_point_pos[1]) or
            towards_east  and game.is_collision(north_point_pos[0], north_point_pos[1]) or
            towards_south and game.is_collision(east_point_pos[0], east_point_pos[1]) or
            towards_west  and game.is_collision(south_point_pos[0], south_point_pos[1]), 
            
            #Move Direction:
            towards_west,
            towards_east, 
            towards_north, 
            towards_south, 
            
            #Food Direction: 
            food_x < head_current_position[0], #food left
            food_x > head_current_position[0], #food right 
            food_y > head_current_position[1], #food up
            food_y < head_current_position[1], #food down      
        ]
        #Check if there is danger 
        return  np.array(state, dtype=int) #in this way i get a boolean matrix of state. 
    
    def remember(self, state, action, reward, next_state, game_is_over):
        
        self.memory.append((state, action, reward, next_state, game_is_over)) #poplef if MAX_MEMORY
       
    
    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:
            
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        
        else: 
            
            mini_sample = self.memory    

        states, actions, rewards, next_states, game_is_ons = zip(*mini_sample)  #This is for getting together all the states, actions, rewards and so on.
        
        self.trainer.train_step(states, actions, rewards, next_states, game_is_ons)

    def train_short_memory(self, state, action, reward, next_state, game_is_on):
        
        self.trainer.train_step(state, action, reward, next_state, game_is_on)
            
    def get_action(self, state):
        
        #random moves: tradeoff exploration / exploitation.
        self.epsilon = FOREPSILON - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1 
            t = "random"

        else: 
            state0 =torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1 
            t = "model"
        return final_move, t

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
        #get  the old state:
        state_old = agent.get_state(game)
        #get_move:
        final_move, t = agent.get_action(state_old)
        #perform move and get new state:
        game_is_over, score, reward = game.play_game(final_move) #review this
        state_new = agent.get_state(game)
        # train short memory 
        
        agent.train_short_memory(state_old, final_move, reward, state_new, game_is_over)
        
        #remember this.
        
        agent.remember(state_old, final_move, reward, state_new, game_is_over)

        if t == "random":
          tr += 1
        else:
          tm += 1      
          
        tt = tr+tm
          
        if game_is_over:
            #train long memory, plot result |
            game.reset_game()
            agent.n_games += 1
            agent.train_long_memory() 
            if score > record: 
                record = score 
                agent.model.save()
                
            print(f"Game: {agent.n_games}. Score: {score}. Record: {record}. %M: {'{:.0%}'.format(tm/tt)}.")    
            tm, tr = 0, 0    
            plot_scores.append(score)
            total_score += score 
            mean_score = total_score / agent.n_games     
            plot_mean_scores.append(mean_score)        
            plot(plot_scores, plot_mean_scores)
            
            

train()   
        
        
