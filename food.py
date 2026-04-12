import random
from turtle import Turtle


class Food(Turtle):
    def __init__(self):
        super().__init__()
        self.shape("circle")
        self.penup()
        self.color("red")
        self.speed("fastest")
        self.refresh()

    def refresh(self):
        random_x = random.randint(0, 280 // 20) * 20
        random_y = random.randint(0, 220 // 20) * 20
        if random.choice([True, False]):
            random_x = -random_x
        if random.choice([True, False]):
            random_y = -random_y
        self.goto(random_x, random_y)
