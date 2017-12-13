import chess.uci

engine = chess.uci.popen_engine("stockfish")
engine.uci()

board = chess.Board("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1")
engine.position(board)
best = engine.go(movetime = 2000)
print(board)
print(best)
