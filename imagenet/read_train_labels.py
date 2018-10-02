
f = open("map_clsloc.txt", 'r')

lines = f.readlines()

labels = {}
for line in lines:
    line = line.split(' ')
    labels[line[0]] = int(line[1])
    
print labels
