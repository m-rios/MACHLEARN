from stockfish import Stockfish
import drawboard

class ChessGame:
    def __init__(self, fen):
        """ constructor taking initial FEN position""" 
        self.d_engine = Stockfish()
        self.setFen(fen)

    def setFen(self, fen):
        """ set board in fen notation """
        self.d_engine.set_fen(fen)

    def move(self, algMove):
        """ move in long algebraic notation, send to stockfish """
        if self.d_engine.is_move_correct(algMove):
            print("correct")

    def bestMove(self):
        return self.d_engine.get_best_move()

if __name__ == "__main__":
    fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    game = ChessGame(fen)
    print(game.bestMove())
    game.move(game.bestMove())

        


    


