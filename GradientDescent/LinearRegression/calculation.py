import numpy as np
from gd import gradientdescent

# Coursera fun Homework problem
x = np.random.rand(5,6)
y = np.array([[0,1,0,1,0]])
data = np.hstack((x, y.T))

w0 = np.random.rand(x.shape[1]).reshape(-1,)-0.5
b0 = 0.5
Lambda = 0.001
epsilon=10**-6
GD = gradientdescent(data, w0, b0, alpha=1*(10**-3),\
                     regularize=False, Lambda=Lambda,\
                     epsilon=epsilon, plot=False, verbose=True)  
## Does not converge
w, b = GD.get_gd()
GD.compare(w, b)
