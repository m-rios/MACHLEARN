import chess
import chess.uci
import sys

handler = chess.uci.InfoHandler()
engine = chess.uci.popen_engine('/home/s3485781/stockfish-8-linux/Linux/stockfish_8_x64') #give correct address of your engine here
engine.info_handlers.append(handler)

if len(sys.argv) < 3:
    print('Must pass args')
    quit() 

_from = int(sys.argv[1])
_to = int(sys.argv[2])

with open('/data/s3485781/datasets/fen_games','r') as f:
    fens=f.readlines()
    with open('/data/s3485781/datasets/labels','r') as label_file:
        labels = label_file.readlines() 
        
        for idx in range(_from-1, _to):
            fen = fens[idx]
            label = labels[idx]
            board = chess.Board(fen)
            engine.position(board)
            evaluation = engine.go(depth=20)
            stock = handler.info["score"][1].cp/100.0

            if stock < 0:
                stock = -1
            elif stock > 0:
                stock = 1
            else:
                stock = 0

            print('line {}: {}'.format(idx+1, int(label) == int(stock)))


#        for i in range(299700,len(lines)):
#            print('iteration {}'.format(i))
#            fen = lines[i]
#            board = chess.Board(fen)
#            engine.position(board)
#            evaluation = engine.go(depth=20)
#            score = handler.info["score"][1].cp/100.0
#            
#            if score < 0:
#                score = -1
#            elif score > 0:
#                score = 1
#            else:
#                score = 0
#            label_file.write('{}\n'.format(score))
#            if not (c % 100):
#                label_file.flush()
#            c += 1

