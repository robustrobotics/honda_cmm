import numpy as np
import matplotlib.pyplot as plt

x = [*range(0, 50, 1)]
train_y = []
val_y = []
files = ['results/results50_1.txt', 'results/results50_2.txt', 'results/results50_3.txt', 'results/results50_4.txt',
         'results/results50_5.txt']
for file in files:
    with open(file) as f:
        lineList = f.readlines()
    i = 0
    while i < 100:
        line = lineList[i][:-1]
        elem = line.split()
        y = float(elem[-1])
        train_y.append(y)
        line = lineList[i + 1][:-1]
        elem = line.split()
        y = float(elem[-1])
        val_y.append(y)
        i += 2

base_train_y = []
base_val_y = []
base_files = ['results/results_b1.txt', 'results/results_b2.txt', 'results/results_b3.txt', 'results/results_b4.txt',
              'results/results_b5.txt']
for filename in base_files:
    with open(filename) as f:
        lineList = f.readlines()
    i = 0
    while i < 100:
        line = lineList[i][:-1]
        elem = line.split()
        y = float(elem[-1])
        base_train_y.append(y)
        line = lineList[i + 1][:-1]
        elem = line.split()
        y = float(elem[-1])
        base_val_y.append(y)
        i += 2

reg_x = [*range(5, 51, 5)]
reg_val_y = []
reg_files = ['results/reg50_1.txt', 'results/reg50_2.txt', 'results/reg50_3.txt', 'results/reg50_4.txt',
             'results/reg50_5.txt']
for filename in reg_files:
    with open(filename) as f:
        lineList = f.readlines()
    i = 0
    while i < 118:
        line = lineList[i][:-1]
        elem = line.split()
        y = float(elem[-1])
        reg_val_y.append(y)
        i += 13

# gp_x = [*range(5, 51, 5)]
# gp_val_y = []
# gp_files = ['results/gpsteps1.txt']
# for filename in gp_files:
#     with open(filename) as f:
#         lineList = f.readlines()
#     i = 0
#     while i < 30:
#         line = lineList[i]
#         elem = line.split()
#         y = float(elem[-1])
#         gp_val_y.append(y)
#         i += 1
#
# g_x = [*range(5, 51, 5)]
# g_val_y = []
# g_files = ['results/gpsteps2.txt']
# for filename in g_files:
#     with open(filename) as f:
#         lineList = f.readlines()
#     i = 0
#     while i < 40:
#         line = lineList[i]
#         elem = line.split()
#         y = float(elem[-1])
#         g_val_y.append(y)
#         i += 1
#
# r_x = [*range(5, 51, 5)]
# r_val_y = []
# r_files = ['results/regsteps1.txt']
# for filename in r_files:
#     with open(filename) as f:
#         lineList = f.readlines()
#     i = 0
#     while i < 40:
#         line = lineList[i]
#         elem = line.split()
#         y = float(elem[-1])
#         r_val_y.append(y)
#         i += 1

x = np.array(x)
y = np.array(val_y).reshape(5, 50)
err_y = np.std(y, axis=0).ravel()
mean_y = np.mean(y, axis=0).ravel()
plt.plot(x, mean_y, color='#713ead', label='Continual w/ buffer')
plt.fill_between(x, mean_y - err_y, mean_y + err_y,
                 alpha=0.5, edgecolor='#713ead', facecolor='#cda3ff')

x = np.array(x)
base_y = np.array(base_val_y).reshape(5, 50)
berr_y = np.std(base_y, axis=0).ravel()
bmean_y = np.mean(base_y, axis=0).ravel()
plt.plot(x, bmean_y, color='#38aba3', label='Continual w/o buffer')
plt.fill_between(x, bmean_y - berr_y, bmean_y + berr_y,
                 alpha=0.2, edgecolor='#38aba3', facecolor='#73f0e7')

x = np.array(reg_x)
reg_y = np.array(reg_val_y).reshape(5, 10)
rerr_y = np.std(reg_y, axis=0).ravel()
rmean_y = np.mean(reg_y, axis=0).ravel()
plt.plot(x, rmean_y, color='#448554', label='Original')
plt.fill_between(x, rmean_y - rerr_y, rmean_y + rerr_y,
                 alpha=0.2, edgecolor='#448554', facecolor='#80c291')

# x = np.array(gp_x)
# y = np.array(gp_val_y).reshape(3, 10)
# err_y = np.std(y, axis=0).ravel()
# mean_y = np.mean(y, axis=0).ravel()
# plt.plot(x, mean_y, color='#b53b33', label='Continual w/ continual exploration')
# plt.fill_between(x, mean_y - err_y, mean_y + err_y,
#                  alpha=0.5, edgecolor='#b53b33', facecolor='#ed908a')
#
# x = np.array(g_x)
# y = np.array(g_val_y).reshape(4, 10)
# err_y = np.std(y, axis=0).ravel()
# mean_y = np.mean(y, axis=0).ravel()
# plt.plot(x, mean_y, color='#4071a3', label='Continual w/o exploration')
# plt.fill_between(x, mean_y - err_y, mean_y + err_y,
#                  alpha=0.5, edgecolor='#4071a3', facecolor='#8ac4ff')
#
# x = np.array(r_x)
# reg_y = np.array(r_val_y).reshape(4, 10)
# rerr_y = np.std(reg_y, axis=0).ravel()
# rmean_y = np.mean(reg_y, axis=0).ravel()
# plt.plot(x, rmean_y, color='#448554', label='Original')
# plt.fill_between(x, rmean_y - rerr_y, rmean_y + rerr_y,
#                  alpha=0.2, edgecolor='#448554', facecolor='#80c291')

plt.legend()
plt.xlabel("L")
plt.ylabel("Validation Loss")
plt.title("Evaluated on N=20 Sliders")
plt.show()
# plt.savefig('val.png')
