import numpy as npy

def expandEmpty(char):

    """ expand empty places to spaces in fen """ 

    if char.isnumeric():
        piece = int(char) * "."
    else:
        piece = char
    return piece

def parsePieces(fen):

    """ get piece info from fen string """ 

    split = fen.split('/', 8)               # split ranks
    split[7] = split[7].split(' ')[0]       # remove extra info (?)
    parsedfen = ""                          # insert empty space
    for rank in split:
        for char in rank:
            parsedfen += expandEmpty(char)
    return parsedfen

def findPieces(parsedfen):

    """ create a position string for each piece """

    out = [0] * 64 * 4
    out[parsedfen.index('K')]             = 1             # white king
    out[parsedfen.index('k') + 64]        = 1             # black king
    out[parsedfen.index('N') + 2 * 64]    = 1             # white knight
    out[parsedfen.index('r') + 3 * 64]    = 1             # black night
    return out

def fromFen(fenstring):

    """ convert fen notaton to bit notation """ 

    return findPieces(parsePieces(fenstring))




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

