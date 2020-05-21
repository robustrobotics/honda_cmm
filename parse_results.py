import numpy as np
import matplotlib.pyplot as plt

x = [*range(0, 50, 1)]
train_y = []
val_y = [] 
files = ['results50_1.txt', 'results50_2.txt', 'results50_3.txt', 'results50_4.txt', 'results50_5.txt']
for file in files:
	with open(file) as f:
  		lineList = f.readlines()
	i = 0
	while i < 100:
		line = lineList[i][:-1]
		elem = line.split()
		y = float(elem[-1])
		train_y.append(y)
		line = lineList[i+1][:-1]
		elem = line.split()
		y = float(elem[-1])
		val_y.append(y)
		i += 2

base_train_y = [] 
base_val_y = []
base_files = ['results_b1.txt', 'results_b2.txt', 'results_b3.txt', 'results_b4.txt', 'results_b5.txt']
for filename in base_files:
	with open(filename) as f:
  		lineList = f.readlines()
	i = 0
	while i < 100:
		line = lineList[i][:-1]
		elem = line.split()
		y = float(elem[-1])
		base_train_y.append(y)
		line = lineList[i+1][:-1]
		elem = line.split()
		y = float(elem[-1])
		base_val_y.append(y)
		i += 2

x = np.array(x)
y = np.array(val_y).reshape(5,50)
err_y = np.std(y, axis = 0).ravel()
mean_y = np.mean(y, axis = 0).ravel()
plt.plot(x, mean_y, color='#b53b33', label = 'with buffer')
plt.fill_between(x, mean_y-err_y, mean_y+err_y,
    alpha=0.5, edgecolor='#b53b33', facecolor='#ed908a')

x = np.array(x)
base_y = np.array(base_val_y).reshape(5,50)
berr_y = np.std(base_y, axis = 0).ravel()
bmean_y = np.mean(base_y, axis = 0).ravel()
plt.plot(x, bmean_y, color='#1B2ACC', label = 'without buffer')
plt.fill_between(x, bmean_y-berr_y, bmean_y+berr_y,
    alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')
plt.legend()
# plt.show()
plt.savefig('validation.png')