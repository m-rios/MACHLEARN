from agent import Agent
# from stockfish import Stockfish
import chess

class StockAgent( Agent ):

    """StockAgent
    
    Agent powered by Stockfish. Can be used to train RL algorithms against it.
    """


    def __init__(self, state, movetime):
        super(StockAgent, self).__init__(state)
        self.board = chess.Board(state)
        self.handler = chess.uci.InfoHandler()
        self.engine = chess.uci.popen_engine("stockfish")
        self.movetime = movetime


    def next_action(self):
        # Reinitialize Stockfish with new State. This might have to be modified 
        # once we settle on a representation system other than moves history
        self.board = chess.Board(self.state)
        self.engine.position(self.board)
        (move, _) = self.engine.go(movetime=self.movetime)
        return move

