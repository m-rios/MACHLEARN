from agent import Agent
from stockfish import Stockfish

class StockAgent( Agent ):

    """StockAgent
    
    Agent powered by Stockfish. Can be used to train RL algorithms against it.
    """


    def __init__(self, state):
        super(StockAgent, self).__init__(state)
        self.stockfish = Stockfish()

    def next_action(self):
        # Reinitialize Stockfish with new State. This might have to be modified 
        # once we settle on a representation system other than moves history
        self.stockfish._Stockfish__start_new_game()
        self.stockfish.set_position(self.state)
        return self.stockfish.get_best_move()