#-----------------------------------|
# Email: indrajit.maity02@gmail.com |
# Author: Indrajit Maity            |
#-----------------------------------|


import numpy as np
import matplotlib.pyplot as plt
import sys


class gradientdescent(object):
  """
  Multiple variable logistic regression
  This code is written while learning Supervised Machine Learning. 
  """
  def __init__(self, data, w0, b0, alpha=0.01, epsilon=10**-10,\
               numiter=100000, plot=True, verbose=False):
    """
    Initialize the attributes
    @input
      data: Data containing both x, y for the linear regression
            No default
            x can be a vector and y is an array. 
            Overall, they have mxn dimension; where m is number of
            samples, and n-1 is number of input variables.
            Data structure
            --------------
            mxn dimension, overall.
            
            For example, 0-th row has x1[0], x2[0],.., and y[0]. 

      w0, b0: Initial guess; 
              Intentionally no defaults; 
      alpha: learning rate; very much depends on gradient
             Defaults to 0.01 
      epsilon: Convergence check over cost function.
             Defaults to 10^-4
      numiter: Number of steps if not converged
             Defaults to 10000
      verbose: How much to print
    """
    self.data = data
    self.w0 = w0
    self.b0 = b0
    self.alpha = alpha
    self.epsilon = epsilon
    self.numiter = numiter 
    self.verbose = verbose
    self.plot = plot
    if self.verbose == True:
      print()
      print(f"-"*60)
      print(f"Gradient Descent implementation with multiple input variables")
      print(f"Logistic Regression used for Classifications")
      print(f"-"*60)
      print()
      print(f"x data: {self.data[:,:-1]}")
      print(f"y data: {self.data[:,-1]}")
      print(f"Initial w0:{self.w0}")
      print(f"Initial b0:{self.b0}")
      print(f"Learning rate (alpha): {self.alpha}")
      print(f"Convergence param. (epsilon): {self.epsilon}")
      print(f"Max. iterations (numiter): {self.numiter}")
      print()


  def get_predict(self, w, b):
    """
    Based on the set of parameters predict the y-values
    using Logistic Regression model
    @input
      w: Number of parameters 
         (n-1) x 1 dimensional array; See __init__ for def. of n;
      b: scalar number
      NOTE: (w,b) makes the full parameter list
    """
    # Logistic: f_wb = 1/(1+e^-z) where z = w.x + b
    # Number of rows, m is the number of training samples
    m = self.data.shape[0]
    # w is n-1 dimensional and x is (m, n-1) dimensional
    z = (np.dot(self.data[:,:-1], w) + b)
    return 1/(1+np.exp(-z))


  def get_cost(self, w, b):
    """
    Compute the cost function based on the parameters
    See get_predict function for details of the input
    Definition (Note that it's not squared error):
      J = (1/m)* \sum_{i} L_{i} where
      L_i = -log(f_i)  if i = 1
      L_i = -log(1-f_i) if i = 0 
    """
    # For all the samples do the following
    # Note that we are using all the samples
    # L is a loss function
    L = 0.0
    # f is 1xm dimensional array;m= number of samples.
    f = self.get_predict(w, b)
    for i in range(self.data.shape[0]):
      L = L - (self.data[i,-1]*np.log(f[i])) -\
              ((1-self.data[i,-1])*np.log(1-f[i]))
    return L/self.data.shape[0]


  def get_gradient(self, w, b):
    """
    Compute Gradient dJ_dw and dJ_db for the gradient descent
    optimization where the function is obtained with logistic
    regression.

    See get_predict function for details of the input
    Definition:
      if j <= 0 -> n-2 [or, dJ_dw]
        dJdj = (1/m)*\sum_{i} [(f^{i} - y^{i}).x^{i}] for every j
      elif j == n-1: [or, dJ_db]
        dJdj = (1/m)*\sum_{i} (f^{i} - y^{i})
    """
    # Note that m is the number of samples
    m,n = self.data.shape
    
    dJdj = np.zeros((n), dtype=float)
    # diff is a mx1 dimensional array
    diff = np.subtract(self.get_predict(w, b), self.data[:,-1])
    # number of parameters 
    for a in range(n):
      if a == n-1:
        dJdj[-1] = np.sum(diff[:])/m
      else:
        tmp =0.0
        # number of samples
        for i in range(m):
          tmp = tmp + (diff[i]*self.data[i,a])
        dJdj[a] = tmp/m
    return dJdj

  
  def compare(self, w, b):
    """
    At any stage of the calculations you may choose to compare
    visually for faster debugging
    """
    # one variable check
    plt.plot(self.data[:,0], self.data[:,-1], color="b",\
             marker="o", label="Data")
    f = self.get_predict(w,b)
    plt.plot(self.data[:,0], f[:], color="r",\
             marker="o", label="Fitted")
    plt.legend()
    plt.savefig("Comparison.png", dpi=200, bbox_inches="tight",\
                pad_inches=0.1, transparent=True)
    plt.show()



  def get_gd(self):
    """
    Optimize w and b with Gradient descent
    """
    w = self.w0; b = self.b0
    cost_prev = 0.0
    for i in range(self.numiter):
      # Contains all the gradients
      dJdj = self.get_gradient(w, b)
      w = w - self.alpha*dJdj[:-1]
      b = b - self.alpha*dJdj[-1]
      cost_now = self.get_cost(w, b)
      #print("w, b", w,b)
      #print("cost associated", cost_now)
      # Compare visually
      if self.plot == True:
        self.compare(w, b)

      if i != 0 and i != self.numiter-1: 
        # Safety checks for minimization
        if cost_now > cost_prev:
          print(f"Cost function increased at current step!")
          print(f"Cost at {i-1}-th step: {cost_prev}")
          print(f"Cost at {i}-th step: {cost_now}")
          print(f"Exiting...")
          sys.exit()
        # Convergence criteria
        elif np.abs(cost_now-cost_prev) < self.epsilon:
          print(f"Converged after {i+1} iterations")
          break
      elif i == self.numiter-1:
        print("Reached the maximum number of iterations!")
        print("Didn't converge fully")
      cost_prev = cost_now
      if i%100 == 0: 
        print(f"Completed {i}-th iterations")
        print(f"Cost at {i}-th iteration is {cost_prev}")
    return w, b
