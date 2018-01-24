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

def benchmark(agent1, agent2, fens):
    
    handler = chess.uci.InfoHandler()
    engine = chess.uci.popen_engine('stockfish') #give correct address of your engine here
    # engine = chess.uci.popen_engine('/home/s3485781/stockfish-8-linux/Linux/stockfish_8_x64') #give correct address of your engine here
    engine.info_handlers.append(handler)

    # Check input
    assert(isinstance(fens, list) and len(fens) > 0 and isinstance(fens[0],str))
    assert(isinstance(agent1, Agent) and isinstance(agent2, Agent))
    
    #Statistics to be computed for 1st agent
    improve = 0
    deprove = 0
    advantage_kept = 0
    for fen in fens:
        
        board = chess.Board(fen)
        initial_white_to_play = np.sign(board.turn-.5)
        
        #Find advantage at the beginning
        engine.position(board)
        evaluation = engine.go(depth=10)
        handler_score = handler.info["score"][1]
        assert handler_score.cp is not None or handler_score is not None
        if handler_score.cp is not None:
            initial_score = handler_score.cp
        else:
            initial_score = handler_score.mate
            if initial_score > 0:
                initial_score =  100000 - initial_score 
            else: 
                initial_score =  -100000 + initial_score
        

        while not board.is_game_over():
            move = agent1.next_action(board)
            board.push(move)
            if board.is_game_over(): break
            move = agent2.next_action(board)
            board.push(move)

        #Find advantage at the end
        engine.position(board)
        evaluation = engine.go(depth=10)
        handler_score = handler.info["score"][1]
        if handler_score.cp is not None:
            final_score = handler_score.cp
            
        else:
            final_score = handler_score.mate
            if final_score > 0:
                final_score =  100000 - final_score 
            else: 
                final_score =  -100000 + final_score
        

        finish_white_to_play = np.sign(board.turn-.5)

        final_score = final_score * finish_white_to_play * initial_white_to_play
        
        if final_score > initial_score:
            improve += 1
        elif final_score < initial_score:
            deprove += 1
        elif final_score > 0:
            advantage_kept += 1
    
    improve = improve / len(fens)
    deprove = deprove / len(fens)
    advantage_kept = advantage_kept / len(fens)
    
    return improve, deprove, advantage_kept

if __name__ == '__main__':
    
    with open('../data/fen_games') as f:
        fens = f.readlines()

        improve, deprove, advantage_kept = benchmark(StockAgent(), StockAgent(), fens)

        print('improve: {}, deprove: {}, advantage_kept: {}'.format(improve, deprove, advantage_kept))
