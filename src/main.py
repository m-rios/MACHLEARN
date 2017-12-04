from stock_agent import StockAgent
from dojo import Dojo

def main():
    
    dojo = Dojo(StockAgent, StockAgent, [])
    print dojo.train()


if __name__ == '__main__':
    main()