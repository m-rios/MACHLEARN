from agent import Agent
from random import choice

class RandomPlayer( Agent ):
    
    def __init__(self):
        super()
    
    def next_action(self, board):
        return choice(list(board.legal_moves))