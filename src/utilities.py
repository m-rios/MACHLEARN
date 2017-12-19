from random import randint

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
    out[parsedfen.index('r') + 3 * 64]    = 1             # black rook
    return out

def fromFen(fenstring):

    """ convert fen notaton to bit notation """ 

    return findPieces(parsePieces(fenstring))

def toFen(bitmap):
    """toFen
    
    Convert bit notation to fen notation
    
    Arguments:
        bitmap {list} -- bit notation 64*4 with board for each piece
    """
    
    # Split board into pieces representation
    (K, k, N, r) = (bitmap[piece*64:piece*64+64] for piece in range(4))
    
    simplified_board = ['.'] * 64;

    simplified_board[K.index(1)] = 'K'
    simplified_board[k.index(1)] = 'k'
    simplified_board[N.index(1)] = 'N'
    simplified_board[r.index(1)] = 'r'

    fen = ""
    ind = 0
    blank = 0
    for ind in range(64):
        if simplified_board[ind] != '.':
            if blank != 0:
                fen += str(blank)
            fen += simplified_board[ind]
            blank = 0
        else:
            blank += 1

        if ind % 8 == 7:
            if blank > 0:
                fen += str(blank)
            if ind < 63:
            fen+='/'
            blank = 0
    
    fen += ' {} - - 0 1'.format(['w','b'][randint(0,1)])

    return fen

def generate_starting_positions(n=100, n_pieces=4, to_file=False):
    positions = []
    f = open('data/fen_games', 'w')
    for _ in range(n):
        # generate a new position
        pos = generate_position(n_pieces)
        while pos in positions: pos = generate_position(n_pieces)
        positions.append(pos)
        if to_file:
            f.write(pos+'\n')
    f.close()
    return positions

def generate_position(n_pieces, to_file=False):
    pieces = []
    # Place pieces in non overlapping possitions
    for _ in range(n_pieces):
        piece = "{0:b}".format(pow(2,randint(0,63))).zfill(64)
        while  piece in pieces: piece = "{0:b}".format(pow(2,randint(0,63))).zfill(64)
        pieces.append(piece)
    return toFen(list(map(int, list(''.join(pieces)))))

def extract_features(fen):
    """extracts features from fen position
    
    Global Features:
        [0] Side to move {0,1} = {b,w}
        [1:2] Material configuration [N,r] = {0,1} (number of pieces of each type)
    Piece-Centric Features:
        [3:10] Piece list and locations [Kexists, Kpos, Nexists, Npos ...]. Exists = {0,1}. Pos = [0,63]
        [11:14] Mobility of Rook [N,S,W,E] = [0,7]
    Square-Centric Features:
        []
    
    Arguments:
        fen {String} -- fen encoding of the board state
    
    Returns:
        [list of int] -- feature representation of the board
    """
    
    metadata = fen.split('/')[7].split(' ')[1:-1]
    board = parsePieces(fen)
    
    # Global features
    features = []
    features.append(int(metadata[0] == 'w'))
    features.append(int(fen.find('N') >= 0))
    features.append(int(fen.find('r') >= 0))
    # Piece-Centric features
    features.append(int(fen.find('K')))
    features.append(board.find('K'))
    features.append(features[1])
    features.append(board.find('N'))
    features.append(int(fen.find('k')))
    features.append(board.find('k'))
    features.append(features[2])
    features.append(board.find('r'))
    #W,E mobility of the rook
    row = int(features[10]/8) #Actual row of the rook
    square = row * 8
    w = 0
    e = 0
    r_found = False
    for i in range(square, square+8): #traverse row counting free squares
        r_found = (board[i] == 'r') or r_found
        if not r_found:
            w = (w + (board[i] == '.'))*(board[i] == '.')
        else:
            if board[i] == 'r':
                continue
            if board[i] != '.' and board[i] != 'r':
                break
            e += 1
    #N,S mobility of the rook
    column = features[10] % 8
    n = 0
    s = 0
    r_found = False
    for i in range(column, column+64, 8): #traverse column counting free squares
        r_found = (board[i] == 'r') or r_found
        if not r_found:
            n = (n + (board[i] == '.'))*(board[i] == '.')
        else:
            if board[i] == 'r':
                continue 
            if board[i] != '.' and board[i] != 'r':
                break           
            s += 1
    features += [n,s,e,w]
    # Square-Centric features

    return features
    

# if __name__ == '__main__':
#     positions = generate_starting_positions(n=100);
#     print('bye')
        
if __name__ == "__main__":
    test = "rnbqkbnr/ppp2ppp/3p1p2/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(fromFen(test))
