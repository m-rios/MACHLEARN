from numpy import random as npy.random
from numpy import tanh as npy.tanh
from utilities import expandEmpty, parsePieces, findPieces, fromFen

class Layer:

    """ a single layer of the multi-layer perceptron """ 

    def __init__(self, number, inputsize = 1, outputsize = 1): 

        """ constructor """

        self.number     = number 
        self.inputsize  = inputsize
        self.outputsize = outputsize
        self.w          = npy.random.rand(self.inputsize, self.outputsize)

class MultiLayerPerceptron:

    """ a multilayer perceptron implementation """ 

    def __init__(self, eta, numlayers, inputsizes, outputsizes):

        """ constructor """ 

        self.layers    = [Layer(i, inputsizes[i], outputsizes[i])
                                            for i in range(0, numlayers)] 
        self.eta       = eta

    def updateLayer(self, layer):

        """ update layer according to update rule """

        for h in range(1, layer.inputsize):
            for l in range(1, layer.outputsize):
                delta          = self.delta[layer.number][l]
                out            = self.out[layer.number - 1][h] 
                layer.w[h, l] += self.eta * delta * out
        
    def train(self, fenboard, score):

        """ train to produce score given board """

        self.board   = fromFen(fenboard)
        self.targ    = score
        
        self.delta   = calculateDelta();         
        for layer in self.layers:
            updateLayer(layer)

    def delta(self, n):

        """ recusively calculate delta :) """

        if self.delta[n] == None:
            delta(n - 1)
        self.delta[n] = 

    def calculateDelta(self):

        """ calculate a list of delta lists, back-propagating """

        self.delta      = [ [None] * layer.outputsize for layer in self.layers ] 
        self.delta[-1]  = self.targ - score()
        for revidx in range(0, len(self.layers):        # test this
            idx = len(self.layers)  - revidx
            leftsum = 1
            rightsum = 1
            self.delta[idx] = leftsum * activationFunDev(rightsum)

    def activationFun(self, vector):
       
        """ activation function of multilayer perceptron """

        return npy.tanh(vector)

    def activationFunDev(self, vector):
        
        return  1 - npy.tanh(vector) ** 2

    def score(self):

        """ score self.board using the trained algo """

        return 1

    
if __name__ == "__main__":
    test = "rnbqkbnr/ppp2ppp/3p1p2/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # print(fromFen(test))
    
    numlayers = 2
    inputsizes = [64 * 4, 20] 
    outputsizes = [20, 1]
    
    mlp = MultiLayerPerceptron(0.9, numlayers, inputsizes, outputsizes)
    mlp.train(test, 5)

