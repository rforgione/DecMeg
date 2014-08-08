import numpy as np
import matplotlib.pyplot as plt 

a_perf = np.loadtxt(open("alpha_performance.csv", "rb"), delimiter=",")
# sorts the matrix by alpha value
a_perf = a_perf[a_perf[:,0].argsort()]
trainset, = plt.plot(a_perf[:,0], 1-a_perf[:,1], linewidth=2.0)
cv_set, = plt.plot(a_perf[:,0], 1-a_perf[:,2], linewidth=2.0)
plt.legend([trainset, cv_set], ["Training Set", "CV Set"], loc=4)
plt.xlabel("Alpha Value")
plt.ylabel("Error Rate")
plt.title("Error Rate by Alpha Value")
plt.show()