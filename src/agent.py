
class Agent( object ):
    """
    Abstract class defining the common interfaces for all agents, to be used by the Dojo to
    train the algorithms.
    """

    def __init__(self, state=None, fen=None):
        """Constructor
        
        Initializes the Agent. This can be extended by child classes if the algorithms requires
        additional data structures and initialization procedures.
        
        Arguments:
            state {string (for now)} -- Representation of the board configuration (from now it's the history
            of moves, it'll be updated once we decide on a representation method)
        """
        
        self.state = state # Initial representation of the board state.
    
    def next_action(self, board):
        raise NotImplementedError

