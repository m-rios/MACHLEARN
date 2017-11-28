from word2number import w2n

with open('knight-pins', 'r') as dataIn, open('knight-pins-n', 'w') as dataOut:
    for line in dataIn:
        line = dataIn.readline().split(",")

        for i in range(1, 7):
            line[i] = str(w2n.word_to_num(line[i]))
        dataOut.write(','.join(line))
    




