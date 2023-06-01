import numpy as np
from gd import gradientdescent

# Coursera fun Homework problem
x = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([[0, 0, 0, 1, 1, 1]])
data = np.hstack((x, y.T))

w0 = np.zeros((2))
b0 = 0
#w0 = np.array([2,3])
#b0 = 1
numiter=10000
GD = gradientdescent(data, w0, b0, alpha=1*(10**-1),\
                     numiter=numiter, plot=False, verbose=True)  

#print(GD.get_cost(w0, b0))
#print(GD.get_gradient(w0, b0))
w, b = GD.get_gd()
print(w, b)
#GD.compare(w, b)

