import sys 
inn = sys.argv[1]
out = inn + '-unique' 

lines_seen = set() # holds lines already seen
outfile = open(out, "w")
for line in open(inn, "r"):
    if line not in lines_seen: # not a duplicate
        outfile.write(line)
        lines_seen.add(line)
outfile.close()
