from random import randint
import random
import chess
import chess.uci
import numpy as np

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

def findPieces(parsedfen,figure='r'):

    """ create a position string for each piece """

    out = [0] * 64 * 4
    out[parsedfen.index('K')]             = 1             # white king
    out[parsedfen.index('k') + 64]        = 1             # black king
    out[parsedfen.index('N') + 2 * 64]    = 1             # white knight
    out[parsedfen.index(figure) + 3 * 64]    = 1             # black rook
    return out

def fromFen(fenstring, figure='r'):

    """ convert fen notaton to bit notation """ 

    return findPieces(parsePieces(fenstring),figure)

def toFen(bitmap, figure='r'):
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
    simplified_board[r.index(1)] = figure

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

def generate_starting_positions(n=100, n_pieces=4, to_file=False, figure='r'):
    positions = []
    f = open('../data/fen_games', 'w')
    for _ in range(n):
        # generate a new position
        pos = generate_position(n_pieces, figure=figure)
        while pos in positions: pos = generate_position(n_pieces, figure=figure)
        positions.append(pos)
        if to_file:
            f.write(pos+'\n')
    f.close()
    return positions

def generate_position(n_pieces, figure='r'):
    pieces = []
    # Place pieces in non overlapping possitions
    for _ in range(n_pieces):
        piece = "{0:b}".format(pow(2,randint(0,63))).zfill(64)
        while  piece in pieces: piece = "{0:b}".format(pow(2,randint(0,63))).zfill(64)
        pieces.append(piece)
    fen = toFen(list(map(int, list(''.join(pieces)))), figure=figure)

    board = chess.Board(fen)

    while board.is_game_over() or not board.is_valid():
        pieces = []
        # Place pieces in non overlapping possitions
        for _ in range(n_pieces):
            piece = "{0:b}".format(pow(2,randint(0,63))).zfill(64)
            while  piece in pieces: piece = "{0:b}".format(pow(2,randint(0,63))).zfill(64)
            pieces.append(piece)
        fen = toFen(list(map(int, list(''.join(pieces)))), figure=figure)
        board = chess.Board(fen)
    return fen


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
    features.append(int(fen.find('b') >= 0))
    # Piece-Centric features
    features.append(int(fen.find('K')))
    features.append(board.find('K'))
    features.append(features[1])
    features.append(board.find('N'))
    features.append(int(fen.find('k')))
    features.append(board.find('k'))
    features.append(features[2])
    features.append(board.find('b'))

#north east mobility of bishop 

    row = int(features[10]/8) #Actual row of the bishop
    col = int(features[10]%8) #Actual col of the bishop

    ne = 0 
    se = 0 
    nw = 0 
    sw = 0

    r = row 
    c = col
    # north east mobility of bishop
    while(r <= 8 and c <= 8):
        i = 8*r +c
        if board[i]=='.':
            ne +=1 
        else:
            break
        r=r+1 
        c=c+1

    r = row 
    c = col
    # north west mobility of bishop
    while( r <=8 and c >= 0):
        i = 8*r +c
        if board[i]=='.':
            nw +=1 
        else:
            break
        r=r+1 
        c=c-1

    r = row 
    c = col
    # south east mobility of bishop
    while(r >= 0 and c <= 8):
        i = 8*r +c
        if board[i]=='.':
            se +=1 
        else:
            break
        r=r-1 
        c=c+1

    r = row 
    c = col
    # south west mobility of bishop
    while( r >= 0 and c >= 0):
        i = 8*r +c
        if board[i]=='.':
            nw +=1 
        else:
            break
        r=r-1 
        c=c-1

    features += [ne,nw,se,sw]


    # Square-Centric features
    features += attack_defend_maps(fen)
    return features


def attack_defend_maps(fen):
    _map = []
    b = chess.Board(fen)
    for r in range(7,-1,-1):
        for f in range(8):        
            s = chess.square(f, r)
            v = 0 # Initialize to unattacked

            attackers = b.attackers(b.turn, s)

            if len(attackers):
                v = min(map(lambda x: b.piece_at(x).piece_type, attackers))
            _map.append(v)
            
            v = 0 # Initialize to undefended
            defenders = b.attackers(not b.turn, s)
            if len(defenders):
                v = min(map(lambda x: b.piece_at(x).piece_type, defenders))
            _map.append(v)
    return _map

if __name__ == '__main__':
    
    
    # features = extract_features('N7/5K2/8/8/8/8/8/3k3r w - - 0 1')
    # features = extract_features('R7/8/8/7k/8/8/1P6/7K w - - 0 1')
    # features = extract_features('8/1K6/8/8/8/8/8/8 w - - 0 1')
    
    # ad_maps = features[-129:-1];
    # printable_map = np.reshape(ad_maps, (8,16))
    # # print(printable_map)
    # for row in printable_map:
    #     row = row.reshape(8,2)
    #     str_row = map(str, row)
    #     print(' | '.join(str_row))

    pass
    # positions = generate_starting_positions(n=100, to_file=True);
    # print('done')

    fen = '5k2/8/K7/8/8/2N5/8/b7 b - - 0 1'
    #features = extract_features(fen)
    #print(features)
    print(sum(fromFen(fen, figure='b')))

        
# if __name__ == "__main__":
#     test = "rnbqkbnr/ppp2ppp/3p1p2/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#     print(fromFen(test))
