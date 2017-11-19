import random

class RandomPlayer:
    def __init__(self, p = 0.5):
        self.p_defect = p
    def move(self, game):
        return random.uniform(0,1) < self.p_defect
    def record(self, game):
        pass

class SimpleGame:
    def __init__(self, player1, player2, payoffmat):
        self.players = [player1, player2]
        self.playoffmat = payoffmat
        self.history = list()
    def run(self, game_iter = 4):
        player1, player2 = self.players
        for iteration in range(game_iter):
            newmoves = player1.move(self), player2.move(self)
            self.history.append(newmoves)   
        player1.record(self); player2.record(self)
    def payoff(self):
        player1, player2 = self.players
        payoffs = self(payoffmat[m1][m2] for (m1, m2) in self.history)
        pay1, pay2 = transpose(payoffs)
        return {player1:mean(pay1), player2:mean(pay2)}

if __name__ == "__main__":
    PAY = [[(3, 3), (0, 5)], [(5, 0), (1, 1)]]
    player1 = RandomPlayer()
    player2 = RandomPlayer()
    # game
    game = SimpleGame(player1, player2, PAY)
    game.run()
    # results
    payoffs = game.payoff()
    print("Player 1 payoff: ", payoffs[player1])
    print("Player 2 payoff: ", payoffs[player2])

