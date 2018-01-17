import chess
import chess.uci

handler = chess.uci.InfoHandler()
engine = chess.uci.popen_engine('/home/s3485781/stockfish-8-linux/Linux/stockfish_8_x64') #give correct address of your engine here
engine.info_handlers.append(handler)

with open('/data/s3485781/datasets/fen_games') as f:
    lines=f.readlines()
    with open('/data/s3485781/datasets/labels','a') as label_file:
        c = 0
        for i in range(299700,len(lines)):
            print('iteration {}'.format(i))
            fen = lines[i]
            board = chess.Board(fen)
            engine.position(board)
            evaluation = engine.go(depth=20)
            score = handler.info["score"][1].cp/100.0
            
            if score < 0:
                score = -1
            elif score > 0:
                score = 1
            else:
                score = 0
            label_file.write('{}\n'.format(score))
            if not (c % 100):
                label_file.flush()
            c += 1

