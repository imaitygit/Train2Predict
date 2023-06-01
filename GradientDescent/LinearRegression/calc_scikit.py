
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("./data/house.txt", unpack=True, delimiter=",",\
                     skip_header=1).T
gdwithscikit = GDwithscikit(data, verbose=False)
gdwithscikit.compare()
