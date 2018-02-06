import chess
import chess.uci
import numpy as np

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

if __name__ == '__main__':
    
    handler = chess.uci.InfoHandler()
    engine = chess.uci.popen_engine('stockfish') #give correct address of your engine here
    # engine = chess.uci.popen_engine('/home/s3485781/stockfish-8-linux/Linux/stockfish_8_x64') #give correct address of your engine here
    engine.info_handlers.append(handler)
    fen = "rnbqkbnr/ppp2ppp/3p1p2/8/8/8/8/4K3 w KQkq - 0 1"
    engine.position(chess.Board(fen))
    evaluation = engine.go(depth=10)
    handler_score = handler.info["score"][1]
    if handler_score.cp is not None:
        print('Centipawn: {}'.format(np.sign(handler_score.cp)))
    else:
        print('Mate: {}'.format(np.sign(handler_score.mate)))