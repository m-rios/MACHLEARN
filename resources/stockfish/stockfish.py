""" Before running do
    pip install stockfish
    to get the stockfish python interface """

from stockfish import Stockfish

stockfish = Stockfish()
stockfish.set_position(['e2e4', 'e7e6'])
stockfish.depth = 20
print(stockfish.get_best_move())
print(stockfish.is_move_correct('a2a3'))
