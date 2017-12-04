from agent import Agent
from stockfish import Stockfish

class Dojo( object ):
    """Dojo class
    
    Dojo implements the training of the agents. It receives any two agents and has them play
    against each other, taking care of the communication between them and the environment (Stockfish).
    """
    
        
    def __init__(self, white, black, initialState):
        """Constructor
        
        Initialises the Dojo class
        
        Arguments:
            white {Agent} -- Agent class of the first agent to play (white)
            black {Agent} -- Agent class of the second agent to play (black)
            initialState {For now this is a list of strings with each move} -- initial configuration of the board
        """
        
        assert issubclass(white, Agent) and issubclass(black, Agent)

        self.white = white(initialState)
        self.black = black(initialState)
        # Depending on our implementation of the state representation we might have to do some processing to pass it to
        # stockfish, as of now i'll assume it is an string already compatible with Stockfish
        self.stockfish = Stockfish()
        self.stockfish.set_position(initialState)
        self.initialState = initialState
    
    def train(self):
        """Train
        
        Main loop where the game is actually played
        """
        # Until we have a better representation system, state is determined by the history of moves
        state = self.initialState
        while True:
            # White turn
            whiteMove = self.white.next_action()
            # This is a safeguard and should never be happen. Agents should always propose valid moves.
            assert self.stockfish.is_move_correct(whiteMove)
            if whiteMove is None:
                break
            self.stockfish.set_position([whiteMove])
            state += [whiteMove]
            self.black.state = state
            self.black.reward = 0 #Boilerplate until we implement a proper reward system
            # Black turn
            blackMove = self.black.next_action()
            assert self.stockfish.is_move_correct(blackMove)
            if blackMove is None:
                break
            self.stockfish.set_position([blackMove])
            state += [blackMove]
            self.white.state = state
            self.white.reward = 0
        
        return state