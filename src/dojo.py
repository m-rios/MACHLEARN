from agent import Agent
import chess
import chess.uci
import chess.svg

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
            initialState {string} -- initial configuration of the board in FEN notation
        """
        
        assert issubclass(white, Agent) and issubclass(black, Agent)

        self.white = white(initialState, 100)
        self.black = black(initialState, 100)
        # Depending on our implementation of the state representation we might have to do some processing to pass it to
        # stockfish, as of now i'll assume it is an string already compatible with Stockfish
        self.board = chess.Board(initialState)
    
    def play(self):
        """Play
        
        Main loop where the game is actually played

        Returns a history with all the moves in svg.
        """
        history = []
        while True:
            # White turn
            whiteMove = self.white.next_action()
            # This is a safeguard and should never be happen. Agents should always propose valid moves.
            assert whiteMove in self.board.legal_moves
            self.board.push(whiteMove)
            history.append(chess.svg.board(board=self.board))
            if self.board.is_game_over():
                break
            self.black.state = self.board.fen()
            # self.black.reward = engine.
            # Black turn
            blackMove = self.black.next_action()
            assert blackMove in self.board.legal_moves
            self.board.push(blackMove)
            history.append(chess.svg.board(board=self.board))
            if self.board.is_game_over():
                break
            self.white.state = self.board.fen()
            # self.white.reward = self.board.
        
        return (history, self.board.result())