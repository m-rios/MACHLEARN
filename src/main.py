from stock_agent import StockAgent
from dojo import Dojo
import os

def main():
    
    dojo = Dojo(StockAgent, StockAgent, 'n7/5k2/8/8/8/8/8/3R3K w - - 0 1')
    (history, result) = dojo.play()
    
    mydir = 'results/svg/'
    filelist = [ f for f in os.listdir(mydir) if f.endswith(".svg") ]
    for f in filelist:
        os.remove(os.path.join(mydir, f))

    for i in range(len(history)-1):
        with open('{}move_{}.svg'.format(mydir, i), 'w') as f:
            f.write(history[i])
    print(result)

if __name__ == '__main__':
    main()