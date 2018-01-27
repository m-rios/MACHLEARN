from agent import Agent
from random_player import RandomPlayer
from stock_agent import StockAgent
import random as rnd
import types
import chess
import numpy as np

# class Benchmarker ( object ):
    
#     def __init__(self, white, black):
#         pass

def compute_score(engine, handler, board):
    engine.position(board)
    evaluation = engine.go(depth=10)
    handler_score = handler.info["score"][1]
    assert handler_score.cp is not None or handler_score is not None
    if handler_score.cp is not None:
        score = handler_score.cp
    else:
        score = handler_score.mate
        if score > 0:
            score =  100000 - score 
        else: 
            score =  -100000 + score
    return score

def benchmark(agent1, agent2, fens):
    
    handler = chess.uci.InfoHandler()
    engine = chess.uci.popen_engine('stockfish') #give correct address of your engine here
    # engine = chess.uci.popen_engine('/home/s3485781/stockfish-8-linux/Linux/stockfish_8_x64') #give correct address of your engine here
    engine.info_handlers.append(handler)

    # Check input
    assert(isinstance(fens, list) and len(fens) > 0 and isinstance(fens[0],str))
    assert(isinstance(agent1, Agent) and isinstance(agent2, Agent))
    
    #Statistics to be computed for 1st agent
    improves = []
    deproves = []
    for fen in fens:
        
        board = chess.Board(fen)
        initial_white_to_play = np.sign(board.turn-.5)
        
        #Find advantage at the beginning
        
        scores_diff = []
        
        n_moves = 0
        while not board.is_game_over():
            score_before = compute_score(engine, handler, board)
            move = agent1.next_action(board)
            board.push(move)
            score_after = compute_score(engine, handler, board) * -1
            scores_diff.append(score_after - score_before)
            if board.is_game_over(): break
            move = agent2.next_action(board)
            board.push(move)
            n_moves += 1
        
        scores_diff = np.sign(scores_diff)
        improves.append(list(scores_diff).count(1)/n_moves)
        deproves.append(list(scores_diff).count(-1)/n_moves)
    
    avg_improve = np.mean(improves)
    avg_deprove = np.mean(deproves)
    
    return avg_improve, avg_deprove

if __name__ == '__main__':
    
    with open('../data/fen_games') as f:
        fens = f.readlines()

        improve, deprove, advantage_kept = benchmark(StockAgent(), StockAgent(), fens)

        print('improve: {}, deprove: {}, advantage_kept: {}'.format(improve, deprove, advantage_kept))
