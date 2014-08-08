import numpy as np
import matplotlib.pyplot as plt 

C_perf = np.loadtxt(open("svc_cost.csv", "rb"), delimiter=",")
# sorts the matrix by C value
C_perf = C_perf[C_perf[:,0].argsort()]
trainset, = plt.plot(C_perf[:,0], 1-C_perf[:,1], linewidth=2.0)
cv_set, = plt.plot(1-C_perf[:,2], linewidth=2.0)
plt.legend([trainset, cv_set], ["Training Set", "CV Set"], loc=1)
plt.show()