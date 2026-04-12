from turtle import Turtle

FONT = ('Courier', 16, 'bold')
ALIGNMENT = 'center'
MENU_FONT = ('Courier', 18, 'bold')
SMALL_FONT = ('Courier', 12, 'normal')


class ScoreBoard(Turtle):
    def __init__(self):
        super().__init__()
        self.color('white')
        self.score = 0
        self.reward = 0
        self.hideturtle()
        self.penup()

        self.header_writer = Turtle(visible=False)
        self.header_writer.color('white')
        self.header_writer.penup()

        self.status_writer = Turtle(visible=False)
        self.status_writer.color('white')
        self.status_writer.penup()

        self.menu_writer = Turtle(visible=False)
        self.menu_writer.color('white')
        self.menu_writer.penup()

        self.update_scoreboard(record=0)

    def update_scoreboard(self, record=0):
        self.header_writer.clear()
        self.header_writer.goto(0, 210)
        self.header_writer.write(
            f'Score: {self.score} | Record: {record}',
            align=ALIGNMENT,
            font=FONT,
        )

    def update_status(self, n_games=0, epsilon=0.0, mode='TRAIN', mean_score=0.0, loaded=False):
        self.status_writer.clear()
        self.status_writer.goto(0, 185)
        loaded_text = 'Yes' if loaded else 'No'
        self.status_writer.write(
            f'Games: {n_games} | Mean: {mean_score:.2f} | Epsilon: {epsilon:.3f} | Mode: {mode} | Loaded: {loaded_text}',
            align=ALIGNMENT,
            font=SMALL_FONT,
        )

    def increase_score(self, record=0):
        self.score += 1
        self.reward = 10
        self.update_scoreboard(record=record)

    def set_step_reward(self):
        self.reward = 0

    def game_over(self):
        self.reward = -10

    def show_menu(self, lines):
        self.menu_writer.clear()
        y = 70
        for idx, line in enumerate(lines):
            self.menu_writer.goto(0, y - idx * 28)
            font = MENU_FONT if idx == 0 else SMALL_FONT
            self.menu_writer.write(line, align=ALIGNMENT, font=font)

    def clear_menu(self):
        self.menu_writer.clear()
