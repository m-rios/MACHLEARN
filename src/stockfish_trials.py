import chess
import chess.uci
import random as r

with open('../data/fen_games') as f:
    data = f.readlines()

fen = data[r.randint(0,len(data)-1)]

board = chess.Board(fen)

handler = chess.uci.InfoHandler()
engine = chess.uci.popen_engine('stockfish') #give correct address of your engine here
engine.info_handlers.append(handler)

engine.position(board)
evaluation = engine.go(depth=20)

print(fen)

i = 0
for move in evaluation:
    print('Evaluation for move {} = \t {}'.format(board.san(move), handler.info["score"][1].cp/100.0))
    i += 1
