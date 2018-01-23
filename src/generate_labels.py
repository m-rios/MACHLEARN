import chess
import chess.uci
import numpy as np

handler = chess.uci.InfoHandler()
engine = chess.uci.popen_engine('stockfish') #give correct address of your engine here
# engine = chess.uci.popen_engine('/home/s3485781/stockfish-8-linux/Linux/stockfish_8_x64') #give correct address of your engine here
engine.info_handlers.append(handler)

# with open('/data/s3485781/datasets/fen_games') as f:
with open('../data/datasets/fen_games') as f:
    lines=f.readlines()
    # with open('/data/s3485781/datasets/labels','w+') as label_file:
    with open('../data/datasets/labels2','w+') as label_file:
        c = 0
        for i in range(0,len(lines)):
            # print('iteration {}'.format(i))
            fen = lines[i]
            board = chess.Board(fen)
            engine.position(board)
            evaluation = engine.go(depth=10)
            score = handler.info["score"][1]

            assert score.cp is not None or score.mate is not None

            if score.cp is not None:
                score = score.cp
            else:
                score = score.mate
            if score != 0:
                score = np.sign(score)

            label_file.write('{}\n'.format(score))
            if not (c % 100):
                label_file.flush()
            c += 1

