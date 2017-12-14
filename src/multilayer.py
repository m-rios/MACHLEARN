import numpy as npy

from utilities import expandEmpty, parsePieces, findPieces, fromFen

""" ------------------------------------------------------------------------"""
class Layer:

    """ a single layer of the multi-layer perceptron """ 

    def __init__(self, number, inputsize = 1, outputsize = 1): 

        """ constructor """

        self.number     = number                        # number of nodes
        self.inputsize  = inputsize                     
        self.outputsize = outputsize                   
        self.w          = npy.random.rand(self.inputsize, self.outputsize)
        self.deltaw     = npy.zeros((self.inputsize, self.outputsize))
        self.input      = npy.array([0] * inputsize)
        self.output     = npy.array([0] * outputsize)
        self.delta      = npy.array([0] * outputsize)

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

        self.board = fromFen(fenboard)                  # first layer input
        self.targ = score                               # score from stockfish
        
        self.predictScore()                                  # current score
        self.calcDelta()                                     # errors
        self.updateWeights()                                 # update rule

    def calcDelta(self):

        """ calculate a list of delta lists, back-propagating """
       
        self.layers[-1].delta = self.targ - self.layers[-1].output
        for n in [0, 1]:
            for l in range(0, self.layers[n].outputsize):
                print(l)
                print(self.layers[n + 1])
                leftsum  = npy.dot(self.layers[n + 1].w[l, :],
                        self.layers[n + 1].delta)
                rightsum = npy.dot(self.layers[n - 1].output,
                        self.layers[n].w[l, :])
                self.layers[n].delta[l] = leftsum * self.layers[n].actiFun(rightsum)

    def predictScore(self):

        """ get output, forward-propagating """ 

        self.layers[0].input  = self.board 
        self.layers[0].output = self.layers[0].actiFun(npy.dot(self.layers[0].input,
            self.layers[0].w))

        for n in range(1, len(self.layers)):
            self.layers[n].input  = self.layers[n - 1].output 
            self.layers[n].output = self.layers[0].actiFun(npy.dot(self.layers[n].input,
                self.layers[n].w))

    def updateWeights(self):

        """ yeah this updates the weights how did u know """
        
        for n in range(0, len(self.layers)):
            for h in range(0, self.layers[n].inputsize):
                for l in range(0, self.layers[n].outputsize):
                    self.layers[n][h, l] = self.eta * self.layers[n].delta[l] * self.layers[n-1].output[h] 

""" ------------------------------------------------------------------------"""
if __name__ == "__main__":
    test = "rnbqkbnr/ppp2ppp/3p1p2/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # print(fromFen(test))
    
    numlayers = 2
    inputsizes = [64 * 4, 20] 
    outputsizes = [20, 1]
    
    mlp = MultiLayerPerceptron(0.9, numlayers, inputsizes, outputsizes)
    mlp.train(test, 5)
