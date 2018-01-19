from agent import Agent
import chess
import chess.uci

class StockAgent( Agent ):

    """StockAgent
    
    Agent powered by Stockfish. Can be used to train RL algorithms against it.
    """


    def __init__(self, state=None, depth=20):
        super(StockAgent, self).__init__(state)
        self.handler = chess.uci.InfoHandler()
        self.engine = chess.uci.popen_engine("stockfish")
        self.depth = depth


    def next_action(self, board):
        # Reinitialize Stockfish with new State. This might have to be modified 
        # once we settle on a representation system other than moves history
        self.engine.position(board)
        (move, _) = self.engine.go(depth=self.depth)
        return move

