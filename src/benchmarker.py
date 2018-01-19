from agent import Agent
from random_player import RandomPlayer
from stock_agent import StockAgent
import random as rnd
import types
import chess

# class Benchmarker ( object ):
    
#     def __init__(self, white, black):
#         pass

def benchmark(agent1, agent2, fens):
    
    # Check input
    assert(isinstance(fens, list) and len(fens) > 0 and isinstance(fens[0],str))
    assert(isinstance(agent1, Agent) and isinstance(agent2, Agent))
    
    #Statistics to be computed for 1st agent
    wins = 0
    losses = 0
    draws = 0

    for fen in fens:
        
        board = chess.Board(fen)

        agent1_is_white = board.turn

        while not board.is_game_over():
            move = agent1.next_action(board)
            board.push(move)
            if board.is_game_over(): break
            move = agent2.next_action(board)
            board.push(move)
        
        result = board.result()
        print(result)
        if result == '1-0':
            result = 1
        elif result == '0-1':
            result = -1
        else:
            result = 0
        
        if result == 0:
            draws += 1
        elif agent1_is_white:
            if result == 1:
                wins += 1
            else:
                losses += 1
        else:
            if result == 0:
                wins += 1
            else:
                losses += 1
    print((wins, losses, draws))
    return wins, losses, draws

if __name__ == '__main__':
    
    with open('../data/fen_games') as f:
        fens = f.readlines()

        wins, losses, draws = benchmark(StockAgent(), StockAgent(), fens)

        print('wins: {}, losses: {}, draws: {}'.format(wins, losses, draws))
