# ----------------------------------|
# Author: Indrajit Maity            |
# Email: indrajit.maity02@gmail.com |
# ----------------------------------|

# Coffee Roasting example with Tensorflow
# Written the code while solving Coursera 
# homework problems.

# Import stuffs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from cr_aux import *

#-----------
# DATA
#----------
# Load data
x, y = load_coffee_data()
#plot_roast(x, y)
print(f"T Max, Min pre normalization: {np.max(x[:,0]):.6f}, {np.min(x[:,0]):0.6f}")
print(f"time Max, Min pre normalization: {np.max(x[:,1]):.6f}, {np.min(x[:,1]):.6f}")
print()

# Normalization layer 
norm_l = tf.keras.layers.Normalization(axis=-1)
# Learns mean and variance
norm_l.adapt(x)
xn = norm_l(x)
print(f"T Max, Min pre normalization: {np.max(xn[:,0]):.6f}, {np.min(xn[:,0]):0.6f}")
print(f"time Max, Min pre normalization: {np.max(xn[:,1]):.6f}, {np.min(xn[:,1]):.6f}")
print()
# Repeat the data to increase the training size
xtiled = np.tile(xn, (1000,1))
ytiled = np.tile(y, (1000,1))


#---------
# Model
#---------
#Setting global seed: different results for every call to the random op, 
#but the same sequence for every re-run of the program 
tf.random.set_seed(1234)  # applied to achieve consistent results
# Two layers sequenctially
model = Sequential(
  [
      tf.keras.Input(shape=(2,)),
      Dense(3, activation='sigmoid', name = 'layer1'),
      Dense(1, activation='sigmoid', name = 'layer2')
   ]
)
# Take a look at the summary
model.summary()
# Take a look at w, b
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)


# --------------
# Loss function
# -------------
# Loss function and fits
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

# -----------------
# Optimization
#------------------

model.fit(
    xtiled,ytiled,            
    epochs=10,
)
# Set the W, b parameters
model.get_layer("layer1").set_weights([W1,b1])
model.get_layer("layer2").set_weights([W2,b2])


#-------------------
# Prediction
#-------------------
# Test the predictions
x_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
x_testn = norm_l(x_test)
predictions = model.predict(x_testn)

# Converting predictions to probabilities
prob = np.zeros_like(predictions)
for i in range(predictions.shape[0]):
  if predictions[i] >= 0.5:
    prob[i] = 1
  else:
    prob[i] = 0

print(f"decisions = \n{prob}")
print("predictions = \n", predictions)
prob = (predictions >= 0.5).astype(int)
print(f"decisions = \n{prob}")

# plot
#plot_layer(x,y.reshape(-1,),W1,b1,norm_l)
