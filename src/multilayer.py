import numpy as npy

from utilities import expandEmpty, parsePieces, findPieces, fromFen

""" ------------------------------------------------------------------------"""
class Layer:

    """ a single layer of the multi-layer perceptron """ 

    def __init__(self, number, inputsize = 1, outputsize = 1): 

        """ constructor """

        self.number       = number                      # identify layer 
        self.inputsize    = inputsize                     
        self.outputsize   = outputsize                   
        self.w            = npy.asmatrix(npy.random.rand(self.inputsize,
            self.outputsize))
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

    def __init__(self, eta, numlayers, inputsizes, outputsizes):

        """ constructor """ 

        self.eta       = eta
        self.layers    = [Layer(i, inputsizes[i], outputsizes[i])
                                            for i in range(0, numlayers)] 

    def train(self, fenboard, score):

        """ train to produce score given board """

        self.board = npy.transpose(npy.asmatrix(fromFen(fenboard)))                  
        self.targ = score                               # score from stockfish
        
        self.predictScore()                             # current score
        self.calcDelta()                                # delta errors 
        self.updateDeltaWeights()                       # update rule 
        self.updateWeights()                            # apply updates

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
    test = "rnbqkbnr/ppp2ppp/3p1p2/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # print(fromFen(test))
    
    numlayers = 2
    inputsizes = [64 * 4, 20] 
    outputsizes = [20, 1]
    eta = 1/5 
    mlp = MultiLayerPerceptron(eta, numlayers, inputsizes, outputsizes)
    mlp.train(test, 5)
