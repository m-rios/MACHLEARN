def drawpiece(char):
    if char.isnumeric():
        toPrint = int(char) * ".\t"
    else:
        toPrint = char + '\t'
    print(toPrint, end = "")

def drawboard(fenstring):
    split = fenstring.split('/', 8)
    split[7] = split[7].split(' ')[0]
    for rank in split:
        for char in rank:
            drawpiece(char)
        print('\n')
                
if __name__ == "__main__":
    test = "rnbqkbnr/ppp2ppp/3p1p2/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    drawboard(test)
    
