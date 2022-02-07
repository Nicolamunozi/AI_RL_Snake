from turtle import Turtle
import random


class Food(Turtle):
    
    def __init__(self):
        super().__init__()
        self.shape("circle")
        self.penup()
        self.color("red")
        self.speed("fastest")
        self.refresh()       
    
    def refresh(self):
        
        random_x = random.randint(0,280//20)*20
        random_y = random.randint(0,220//20)*20
        sign_x = random.choice([True, False])
        sign_y = random.choice([True, False])
        if sign_x:
            random_x = - random_x
        if sign_y:
            random_y = - random_y
        
        self.goto(random_x,random_y)    
    
    