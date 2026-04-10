from turtle import Turtle

FONT = ("Courier", 20, "bold")
ALIGNMENT = "center"


class ScoreBoard(Turtle):
    def __init__(self):
        super().__init__()
        self.color("white")
        self.score = 0
        self.hideturtle()
        self.penup()
        self.setposition(-40, 200)
        self.update_scoreboard()

    def update_scoreboard(self):
        self.write(f"Score: {self.score}", font=FONT)

    def increase_score(self):
        self.clear()
        self.score += 1
        self.update_scoreboard()

    def game_over(self):
        self.goto(0, 0)
        self.write("GAME OVER", align=ALIGNMENT, font=FONT)
