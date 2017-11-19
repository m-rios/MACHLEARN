import random

class RandomMover:
    def move(self):
        return random.uniform(0,1) < 0.5

PAY = [[(3,3),(0,5)], [(5,0), (1,1)]]
player1 = RandomMover()
player2 = RandomMover()
# get move
move1 = player1.move()
move2 = player2.move()
# get and print payoff
pay1, pay2 = PAY[move1][move2]
print("player1 payoff: ", pay1)
print("player2 payoff: ", pay2)
