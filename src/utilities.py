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
