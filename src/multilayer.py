import numpy as npy
import chess.syzygy
import chess.uci

from utilities import expandEmpty, parsePieces, findPieces, fromFen

""" ------------------------------------------------------------------------"""
class Layer:

    """ a single layer of the multi-layer perceptron """ 

    def __init__(self, number, inputsize = 1, outputsize = 1, iniW = None): 

        """ constructor """

        self.number       = number                      # identify layer 
        self.inputsize    = inputsize                     
        self.outputsize   = outputsize                   

        if iniW == None:
            self.w        = npy.asmatrix(npy.random.rand(self.inputsize,
                                                             self.outputsize))
        else:
            self.w = iniW

        self.deltaw       = npy.asmatrix(npy.zeros((self.inputsize,
                                                    self.outputsize)))
        self.input        = npy.asmatrix(npy.zeros((inputsize, 1)))
        self.output       = npy.asmatrix((npy.zeros((outputsize, 1))))
        self.delta        = npy.asmatrix(npy.zeros((outputsize, 1)))

    @staticmethod                                   
    def actiFun(vector):
       
        """ activation function of the layer """

        return npy.tanh(vector)

    @staticmethod
    def actiFunDev(vector):
        
        """ derivative of activation function """
        
        return  1 - npy.tanh(vector) ** 2

""" ------------------------------------------------------------------------"""
class MultiLayerPerceptron:

    """ a multilayer perceptron implementation """ 

    def __init__(self, eta, numlayers, inputsizes, outputsizes, wList = None):

        """ constructor """ 

        self.eta       = eta
        if wList == None:
            self.layers = [Layer(i, inputsizes[i], outputsizes[i])
                                            for i in range(numlayers)] 
        else:
            self. layers = [layer(i, inputsizes[i], outputsizes[i], wList[i])
                                    for i in range(numlayers)]

    def train(self, fenboard, score):

        """ train to produce score given board """

        self.setboard(fenboard)
        self.targ = score                               # score from stockfish
        
        self.predictScore()                             # current score
        self.calcDelta()                                # delta errors 
        self.updateDeltaWeights()                       # update rule 
        self.updateWeights()                            # apply updates

    def setboard(self, fenboard):

        """ set the board from fen notation """

        self.board = npy.transpose(npy.asmatrix(fromFen(fenboard)))

    def calcDelta(self):

        """ calculate delta's, back-propagating """

        print("calculating delta values...")
       
        self.layers[-1].delta = self.targ - self.layers[-1].output
        for n in range(0, len(self.layers) - 1):
            leftsum  = self.layers[n + 1].w * self.layers[n + 1].delta
            rightsum = npy.transpose(self.layers[n].input) * self.layers[n].w
            self.layers[n].delta = npy.multiply(leftsum,
                    self.layers[n].actiFun(npy.transpose(rightsum)))

    def predictScore(self):

        """ get output, forward-propagating """ 

        print("predicting...")

        self.layers[0].input  = self.board 
        self.layers[0].output = npy.transpose(self.layers[0].actiFun(npy.transpose(self.layers[0].input) *
                self.layers[0].w))

        for n in range(1, len(self.layers)):
            self.layers[n].input  = self.layers[n - 1].output 
            self.layers[n].output = npy.transpose(self.layers[0].actiFun(npy.transpose(self.layers[n].input)
                    * self.layers[n].w))
        return self.layers[-1].output

    def updateDeltaWeights(self):

        """ update deltaw values """

        print("updating deltaw values...")
        
        for layer in self.layers: 
            if layer.input.shape == (1,1):
                layer.input = npy.asscaler(layer.input)
            layer.deltaw += self.eta * layer.input * npy.transpose(layer.delta)

    def updateWeights(self):

        """ apply deltaw to update w """

        print("updating weights...")

        for layer in self.layers:
            layer.w += layer.deltaw
            layer.delta = npy.asmatrix(npy.zeros((layer.inputsize,
                                            layer.outputsize)))


""" ------------------------------------------------------------------------"""
if __name__ == "__main__":

    # construct stockfish
    engine = chess.uci.popen_engine("/home/s1925873/stockfish-8-linux/Linux/stockfish_8_x64")
    engine.uci()
    info_handler = chess.uci.InfoHandler()
    engine.info_handlers.append(info_handler)

   
    # construct mlp     
    numlayers = 2
    inputsizes = [64 * 4, 20] 
    outputsizes = [20, 1]
    eta = 1/5 
    mlp = MultiLayerPerceptron(eta, numlayers, inputsizes, outputsizes)
  
    # read data and train
    data = open("../data/fen_games")
    size = int(data.readline())
    trainSize = int(size * 0.8)
    for _ in range(trainSize):
        fen = data.readline()
        # get stockfish score
        engine.ucinewgame()
        engine.position(chess.Board(fen))
        print(engine.board)
        engine.go(movetime=1000)
        score = info_handler.info["score"][1]
        # training
        print("training on line {}".format(fen))
        mlp.train(fen, score)          # 5 should be score

    # save weights
    wlist = [layer.w for layer in mlp.layers]
    filename = "weights.npz"
    npy.savez(filename, wlist = wlist)

    # read weights
    npzfile = npy.load(filename)
    wlist = npzfile['wlist'] 

    predictor = MultiLayerPerceptron(eta, numlayers, inputsizes, outputsizes)
    for idx, layer in enumerate(predictor.layers):
        layer.w = wlist[idx]

    # calculate error measure 
    E = 0
    score = 5
    fen = data.readline()
    epsilon = 0.1

    while fen:
        predictor.setboard(fen)
        calcScore = predictor.predictScore()
        E += abs(score - calcScore)
        fen = data.readline()

    print("total error: {}".format(E))

    # move predict function
    # introduce proper score
    # 
