import numpy as np
from gd import gradientdescent

# Coursera fun Homework problem
# Prediction of house-prices based on some features
data = np.array([[2104,1416, 852],\
                 [5,3,2],\
                 [1,2,1],\
                 [45,40,35],\
                 [460,232,178]])

# Initial guesses
w0 = np.zeros((4))
b0 = 0.0
GD = gradientdescent(data, w0, b0, alpha=5*(10**-7),\
                     plot=False, verbose=True)  
# Does not converge
w, b = GD.get_gd()
GD.compare(w, b)

