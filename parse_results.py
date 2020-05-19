import numpy as np
import matplotlib.pyplot as plt

with open('results50_1.txt') as f:
  lineList1 = f.readlines()
with open('results50_2.txt') as f:
  lineList2 = f.readlines()
with open('results50_3.txt') as f:
  lineList3 = f.readlines()
with open('results50_4.txt') as f:
  lineList4 = f.readlines()
with open('results50_5.txt') as f:
  lineList5 = f.readlines()

train_x = []
train_y = []
val_y = [] 
i = 0
while i < 100:
	line = lineList1[i][:-1]
	elem = line.split()
	x = float(elem[1][:-1])
	x = int(x)
	y = float(elem[-1])
	train_x.append(x)
	train_y.append(y)
	line = lineList1[i+1][:-1]
	elem = line.split()
	y = float(elem[-1])
	val_y.append(y)
	i+=2

i = 0
while i < 100:
	line = lineList2[i][:-1]
	elem = line.split()
	y = float(elem[-1])
	train_y.append(y)
	line = lineList2[i+1][:-1]
	elem = line.split()
	y = float(elem[-1])
	val_y.append(y)
	i += 2

i = 0
while i < 100:
	line = lineList3[i][:-1]
	elem = line.split()
	y = float(elem[-1])
	train_y.append(y)
	line = lineList3[i+1][:-1]
	elem = line.split()
	y = float(elem[-1])
	val_y.append(y)
	i += 2

i = 0
while i < 100:
	line = lineList4[i][:-1]
	elem = line.split()
	y = float(elem[-1])
	train_y.append(y)
	line = lineList4[i+1][:-1]
	elem = line.split()
	y = float(elem[-1])
	val_y.append(y)
	i += 2

i = 0
while i < 100:
	line = lineList5[i][:-1]
	elem = line.split()
	y = float(elem[-1])
	train_y.append(y)
	line = lineList5[i+1][:-1]
	elem = line.split()
	y = float(elem[-1])
	val_y.append(y)
	i += 2

base_train_y = [] 
base_val_y = []
with open('results_b1.txt') as f:
  lineList1 = f.readlines()
with open('results_b2.txt') as f:
  lineList2 = f.readlines()
with open('results_b3.txt') as f:
  lineList3 = f.readlines()
with open('results_b4.txt') as f:
  lineList4 = f.readlines()
i = 0
while i < 98:
	line = lineList1[i][:-1]
	elem = line.split()
	y = float(elem[-1])
	base_train_y.append(y)
	line = lineList1[i+1][:-1]
	elem = line.split()
	y = float(elem[-1])
	base_val_y.append(y)
	i += 2

i = 0
while i < 98:
	line = lineList2[i][:-1]
	elem = line.split()
	y = float(elem[-1])
	base_train_y.append(y)
	line = lineList2[i+1][:-1]
	elem = line.split()
	y = float(elem[-1])
	base_val_y.append(y)
	i += 2

i = 0
while i < 98:
	line = lineList3[i][:-1]
	elem = line.split()
	y = float(elem[-1])
	base_train_y.append(y)
	line = lineList3[i+1][:-1]
	elem = line.split()
	y = float(elem[-1])
	base_val_y.append(y)
	i += 2

i = 0
while i < 98:
	line = lineList4[i][:-1]
	elem = line.split()
	y = float(elem[-1])
	base_train_y.append(y)
	line = lineList4[i+1][:-1]
	elem = line.split()
	y = float(elem[-1])
	base_val_y.append(y)
	i += 2

x = np.array(train_x)
y = np.array(val_y).reshape(5,50)
err_y = np.std(y, axis = 0).ravel()
mean_y = np.mean(y, axis = 0).ravel()
plt.plot(x, mean_y, color='#b53b33', label = 'with buffer')
plt.fill_between(x, mean_y-err_y, mean_y+err_y,
    alpha=0.5, edgecolor='#b53b33', facecolor='#ed908a')

x = np.array(train_x[:49])
base_y = np.array(base_val_y).reshape(4,49)
berr_y = np.std(base_y, axis = 0).ravel()
bmean_y = np.mean(base_y, axis = 0).ravel()
plt.plot(x, bmean_y, color='#1B2ACC', label = 'without buffer')
plt.fill_between(x, bmean_y-berr_y, bmean_y+berr_y,
    alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')
plt.legend()
plt.savefig('val_loss.png')