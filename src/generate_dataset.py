import utilities as u

positions = u.generate_starting_positions(n=100, figure='b', to_file=True)


for pos in positions:
    print(pos)
