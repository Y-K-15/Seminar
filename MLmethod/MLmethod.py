import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the number of trials and observations
num_trials = 100
num_obs = 1000

# Define alpha, beta, and sigma
alpha = 0
beta = 2
sigma = 400

# Initialize a list to store beta_hat values
beta_hat_values = []

# Define likelihood function for logistic regression
def likelihood(beta_tilda, alpha, x, y, sigma):
    u_tilda = y - (alpha + beta_tilda * x)
    # Probability density for y=1
    prob1 = norm.cdf(u_tilda, 0, sigma)
    # Probability density for y=0
    prob0 = 1 - prob1
    # Total likelihood
    likelihood = np.prod(y*prob1 + (1-y)*prob0)
    # We will use scipy's minimize function which minimizes, so return negative
    return -likelihood

# Run trials
for j in range(num_trials):

    # Generate random component u ~ N(0, sigma)
    u = np.random.normal(0, sigma, num_obs)

    # Create x vector
    x = np.arange(1, num_obs+1)

    # Generate y_star and convert to y as a dummy variable
    y_star = alpha + beta * x + u
    y = (y_star > 0).astype(int)

    # Use scipy's minimize function to find beta_hat that maximizes likelihood
    res = minimize(likelihood, 1, args=(alpha, x, y, sigma), bounds=[(1,3)], method='SLSQP')
    beta_hat = res.x[0]

    # Append the beta_hat value
    beta_hat_values.append(beta_hat)

# Convert list to a DataFrame
df = pd.DataFrame(beta_hat_values, columns=['beta_hat'])

# Plot histogram of beta_hat values
plt.hist(df['beta_hat'], bins=10, edgecolor='k')
plt.xlabel('Beta Hat')
plt.ylabel('Frequency')
plt.title('Histogram of Beta Hat')
plt.show()
