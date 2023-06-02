# A few common functions used for coffee roasting example
import numpy as np
import matplotlib.pyplot as plt

def load_coffee_data():
  """
  NOTE: The roasting duration and temperature range 
        is taken from the Coursera. It just is used 
        as a guide and to crosscheck my 
        implementation. 

  Roasting duration: 12-15 minutes
  T range: 175-260C
  """
  # Random Number Generator
  rng = np.random.default_rng(2)
  # The unspecified value is determined run-time
  X = rng.random(400).reshape(-1,2)
  X[:,1] = X[:,1] * 4 + 11.5          # 12-15 min is best
  X[:,0] = X[:,0] * (285-150) + 150  # 350-500 F (175-260 C) is best
  Y = np.zeros(X.shape[0])
  i=0
  # Two indices 
  for t,d in X:
    y = -3/(260-175)*t + 21
    if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
        Y[i] = 1
    else:
        Y[i] = 0
    i += 1
  return (X, Y.reshape(-1,1))


def plot_roast(x,y):
  """
  Plot Coffee roast data 
  """
  y = y.reshape(-1,)
  colormap = np.array(['r', 'b'])
  fig, ax = plt.subplots(1,1)
  # y == 1 positive outcome and two features are plotted as ^
  ax.scatter(x[y==1,0],x[y==1,1], s=70, marker='^', c='blue', label="Optimal Roast" )
  # y == 1 positive outcome and two features are plotted as X
  ax.scatter(x[y==0,0],x[y==0,1], s=100, marker='X', c="red", label="Bad Roast")
  #tr = np.linspace(175,260,50)
  #ax.plot(tr, (-3/85) * tr + 21, color=dlc["dlpurple"],linewidth=1)
  #ax.axhline(y=12,color=dlc["dlpurple"],linewidth=1)
  #ax.axvline(x=175,color=dlc["dlpurple"],linewidth=1)
  ax.set_title(f"Coffee Roasting", size=16)
  ax.set_xlabel("Temperature \n(Celsius)",size=12)
  ax.set_ylabel("Duration \n(minutes)",size=12)
  ax.legend(loc='upper right')
  plt.show()
