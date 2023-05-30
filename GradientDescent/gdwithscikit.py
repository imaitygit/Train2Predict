# ----------------------------------|
# Author: Indrajit Maity            |
# Email: indrajit.maity02@gmail.com |
# ----------------------------------|

# Inport stuffs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


class GDwithscikit(object):
  """
  Employ scikit-learn to do linear regression with gradient descent
  """
  def __init__(self, data, verbose=True):
    """
    Initialize attributes
    @input
      data: Data containing both x, y for the linear regression
            No default
            x can be a vector and y is an array.
            Overall, they have nxm dimension; where m is number of
            samples, and n-1 is number of input variables.
            -- Data structure --
            First n-1 rows with m columns: x
            nth row with m column: y
    """
    self.data = data
    self.verbose = verbose

    if self.verbose == True:
      print()
      print(f"-"*60)
      print(f"Gradient Descent with scikit-learn")
      print(f"-"*60)
      print()
      print(f"x data: {self.data[:,:-1]}")
      print(f"y data: {self.data[:,-1]}")
      #print(f"Initial w0:{self.w0}")
      #print(f"Initial b0:{self.b0}")
      #print(f"Learning rate (alpha): {self.alpha}")
      #print(f"Convergence param. (epsilon): {self.epsilon}")
      #print(f"Max. iterations (numiter): {self.numiter}")
      print()


  def set_scale(self):
    """
    This function scales the inputs suitably
    x_out = (x - mu)/std where mu and std are the mean and 
                         standard deviation respectively. 
    """
    # fit and transform
    return StandardScaler(with_mean=True, with_std=True).fit_transform(self.data[:,:-1])


  def get_fit(self):
    """
    Fitting with scikit-learn
    """
    # call fit from the sgdr classs
    sgdr = SGDRegressor(loss="squared_error", max_iter=1000)
    sgdr.fit(self.set_scale(), self.data[:,-1])
    return sgdr.intercept_, sgdr.coef_

  def compare(self):
    """
    Compare the data
    """
    b, w = self.get_fit()
    x = self.set_scale()
    f = np.dot(x, w) + b
    plt.scatter(x[:,0], f, color="b", s=8) 
    plt.scatter(x[:,0], self.data[:,-1], color="g", s=8)
    plt.show() 


