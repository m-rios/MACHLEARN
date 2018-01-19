import chess

n = 0
with open('../data/fen_games') as f:
    with open('../data/fen_games_2','w+') as f2:
        for fen in f:
            board = chess.Board(fen)
            if board.is_valid() and not board.is_game_over():
                f2.write(fen)
                n += 1
                if not (n % 1000):
                    f2.flush()
