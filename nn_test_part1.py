#%% Clear variable list
def clearall():
    """clear all globals"""
    for uniquevar in [var for var in globals().copy() if var[0] != "_" and var != 'clearall']:
        del globals()[uniquevar]
clearall()

#%% 
# Python imports
import numpy as np  # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
#import seaborn as sns
# Allow matplotlib to plot inside this notebook
#%matplotlib inline
# Set the seed of the np random number generator so that the tutorial is reproducable
np.random.seed(seed=1)

#%% 
# Define the vector of input samples as x, with 20 values sampled from a uniform distribution between 0 and 1
x = np.random.uniform(0, 1, 20)

# Generate the target values t from x with small gaussian noise so the estimation won't be perfect.
# Define a function f that represents the line that generates t without noise
def f(x): return x * 2

# Create the targets t with some gaussian noise
noise_variance = 0.2  # Variance of the gaussian noise
# Gaussian noise error for each sample in x
noise = np.random.randn(x.shape[0]) * noise_variance
# Create targets t
t = f(x) + noise

# Plot the target t versus the input x
plt.plot(x, t, 'o', label='t')
# Plot the initial line
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
plt.xlabel('$x$', fontsize=15)
plt.ylabel('$t$', fontsize=15)
plt.ylim([0,2])
plt.title('inputs (t) vs targets (x)')
plt.grid()
plt.legend(loc=2)

#%%
# Define the neural network function y = x * w
def nn(x, w): return x * w

# Define the cost function
def cost(y, t): return ((t - y)**2).sum()

# Define a vector of weights for which we want to plot the cost
ws = np.linspace(0, 4, num=100)  # weight values
cost_ws = np.vectorize(lambda w: cost(nn(x, w) , t))(ws)  # cost for each weight in ws

# Plot the cost vs the given weight w
plt.plot(ws, cost_ws, 'r-')
plt.xlabel('$w$', fontsize=15)
plt.ylabel('$cost=SSE$', fontsize=15)
plt.title('cost vs weight')
plt.grid()

#%%
# define the gradient function. Remember that y = nn(x, w) = x * w
def gradient(w, x, t): return 2 * x * (nn(x, w) - t)

# define the update function delta w
def delta_w(w_k, x, t, learning_rate):
    return learning_rate * gradient(w_k, x, t).sum()

# Set the initial weight parameter
w = 0.1
# Set the learning rate
learning_rate = 0.1

# Start performing the gradient descent updates, and print the weights and cost:
nb_of_iterations = 4  # number of gradient descent updates
for i in range(nb_of_iterations):
    # Print the current w, and cost
    print('w({}): {:.4f} \t cost: {:.4f}'.format(i, w, cost(nn(x, w), t)))
    dw = delta_w(w, x, t, learning_rate)  # get the delta w update
    w = w - dw  # update the current weight parameter

# Print the final w, and cost
print('w({}): {:.4f} \t cost: {:.4f}'.format(nb_of_iterations, w, cost(nn(x, w), t)))

#%%
w = 0.1  # Set the initial weight parameter
# Start performing the gradient descent updates, and plot these steps
plt.plot(ws, cost_ws, 'r-')  # Plot the error curve
nb_of_iterations = 2  # number of gradient descent updates
for i in range(nb_of_iterations):
    dw = delta_w(w, x, t, learning_rate)  # get the delta w update
    # Plot the weight-cost value and the line that represents the update
    plt.plot(w, cost(nn(x, w), t), 'bo')  # Plot the weight cost value
    plt.plot([w, w-dw],[cost(nn(x, w), t), cost(nn(x, w-dw), t)], 'b-')
    plt.text(w, cost(nn(x, w), t)+0.5, '$w({})$'.format(i))   
    w = w - dw  # update the current weight parameter
    
# Plot the last weight, axis, and show figure
plt.plot(w, cost(nn(x, w), t), 'bo')
plt.text(w, cost(nn(x, w), t)+0.5, '$w({})$'.format(nb_of_iterations))  
plt.xlabel('$w$', fontsize=15)
plt.ylabel('$cost=SSE$', fontsize=15)
plt.title('Gradient descent updates plotted on cost function')
plt.grid()

#%%
w = 0
# Start performing the gradient descent updates, and print the weights and cost:
nb_of_iterations = 10  # number of gradient descent updates
for i in range(nb_of_iterations):
    dw = delta_w(w, x, t, learning_rate)  # get the delta w update
    w = w - dw  # update the current weight parameter

# Plot the target t versus the input x
plt.plot(x, t, 'o', label='t')
# Plot the initial line
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
# plot the fitted line
plt.plot([0, 1], [0*w, 1*w], 'r-', label='fitted line')
plt.xlabel('input x')
plt.ylabel('target t')
plt.ylim([0,2])
plt.title('input vs target')
plt.grid()
plt.legend(loc=2)
plt.show()

#%%
# Define the logistic function
def logistic(z): return 1 / (1 + np.exp(-z))

# Plot the logistic function
z = np.linspace(-6,6,100)
plt.plot(z, logistic(z), 'b-')
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$\sigma(z)$', fontsize=15)
plt.title('logistic function')
plt.grid()
plt.show()

#%%
# Define the logistic function
def logistic_derivative(z): return logistic(z) * (1 - logistic(z))

# Plot the logistic function
z = np.linspace(-6,6,100)
plt.plot(z, logistic_derivative(z), 'r-')
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$\\frac{\\partial \\sigma(z)}{\\partial z}$', fontsize=15)
plt.title('derivative of the logistic function')
plt.grid()
plt.show()

#%%
