import numpy as npy
from utilities import expandEmpty, parsePieces, findPieces, fromFen

class Layer:

    """ a single layer of the multi-layer perceptron """ 

    def __init__(self, inputsize = 1, outputsize = 1): 
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.w = npy.random.rand(self.inputsize, self.outputsize)

class MultiLayerPerceptron:

    """ a multilayer perceptron implementation """ 

    def __init__(self, eta, numlayers, inputsizes, outputsizes):

        self.layers = [Layer(inputsizes[i], outputsizes[i])
                for i in range(0, numlayers)] 

        self.eta       = eta
        self.numlayers = numlayers                          

    def updateNotFirst(self, layernumber):

        """ update rule for layers 2,3, ... """

        print("UPDATING LAYER NOT FIRST")
    
    def updateFirst(self):

        """ updaterule for layer 1 """ 

        print("UPDATING FIRST LAYER")

    def updateLayer(self, layernumber):

        """ update according to update rule """

        if layernumber >= 1:
            self.updateNotFirst(layernumber)
        else:
            self.updateFirst()
        
    def train(self, fenboard, score):

        """ train to produce score given board """

        self.board = fromFen(fenboard)
        self.targ = score
        for layernum in range(self.numlayers):
            self.updateLayer(layernum)
    
if __name__ == "__main__":
    test = "rnbqkbnr/ppp2ppp/3p1p2/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # print(fromFen(test))
    
    numlayers = 2
    inputsizes = [64 * 4, 20] 
    outputsizes = [20, 1]
    
    mlp = MultiLayerPerceptron(0.9, numlayers, inputsizes, outputsizes)
    mlp.train(test, 5)

