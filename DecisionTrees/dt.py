import numpy as np
import matplotlib.pyplot as plt


class DecisionTree(object):
  """
  Decision Tree example
  """
  def __init__(self, x_train, y_train, verbose=True):
    """
    Initialise attributes
    """
    self.x_train = x_train
    self.y_train = y_train
    self.verbose = verbose

  def entropy(self, p):
    """
    Entropy based on a given p
    To avoid log2(0), we used 0.0
    """
    if p == 0 or p == 1:
      return 0.0
    else:
      return -p*np.log2(p) - (1-p)*np.log2(1-p)


  def split_ind(self, x_ind):
    """
    Given a x_ind feature returns two lists for the two 
    split nodes, the left node has the animals that have 
    that feature = 1 and the right node those that have 
    the feature = 0 
    x_ind = 0 => Ear shape
    x_ind = 1 => Face shape
    x_ind = 2 => Whiskers
    """
    left_ind = []
    right_ind = []
    for i,j in enumerate(self.x_train):
        if j[x_ind] == 1:
            left_ind.append(i)
        else:
            right_ind.append(i)
    return np.array(left_ind), np.array(right_ind)


  def weighted_entropy(self, left_ind,right_ind):
    """
    Based on the split data, the indices we chose to split 
    and returns the weighted entropy.
    """
    # Probability of animals in left and right node
    w_left = left_ind.shape[0]/self.x_train.shape[0]
    w_right = right_ind.shape[0]/self.x_train.shape[0]
    # probability of animals being cat in left and right node
    p_left = np.sum(self.y_train[left_ind])/left_ind.shape[0]
    p_right = np.sum(self.y_train[right_ind])/right_ind.shape[0]
    weighted_entropy = w_left * self.entropy(p_left) +\
                       w_right * self.entropy(p_right)
    return weighted_entropy


  def information_gain(self, left_ind, right_ind):
    """
    returns the entropy gain relative to the start
    """
    p_node = np.sum(self.y_train)/self.y_train.shape[0]
    h_node = self.entropy(p_node)
    w_entropy = self.weighted_entropy(left_ind,right_ind)
    # Gain
    return h_node - w_entropy

 
  def recursion(self):
    """
    Recursive
    """
    for i, feature in enumerate(['Ear', 'Face', 'Whiskers']):
      left_ind, right_ind = self.split_ind(i)
      i_gain = self.information_gain(left_ind, right_ind)
      if self.verbose == True:
        print(f"Feature: {feature}, information gain using this feature: {i_gain:.4f}")


# Data
# Ear shape:: 1-> pointy, 0-> floppy
# Face shape:: 1-> Round, 0-> Not round
# Whiskers:: 1-> Present, 0-> Absent
x_train = np.array([
 [1, 1, 1],
 [0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])
y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

DT = DecisionTree(x_train, y_train)
print(DT.recursion())

