with open('results50_1.txt') as f:
  lineList = f.readlines()

for line in lineList:
	line = line[:-1]
	elem = line.split()
	x = float(elem[1][:-1])
	x = int(x)
	y = float(elem[-1])
	print(elem)
	print(x)
	print(y)
	break 