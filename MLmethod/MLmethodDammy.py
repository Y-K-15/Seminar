import numpy as np
import pandas as pd
from scipy.stats import norm
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

# Define log likelihood function
def log_likelihood(beta_tilda, alpha, x, y, sigma):
    # prob(y_i=0)
    # =prob(y*_i <= 300)
    # =prob(alpha + beta_tilda * x + u_i <= 300)
    # =prob(ui <= 300 - alpha - beta_tilda* x)
    
    # Probability density for y=0
    prob0 = norm.cdf(300 - alpha - beta_tilda* x, 0, sigma)
    # Probability density for y=1
    prob1 = 1 - prob0
    # Total log likelihood
    log_likelihood = np.sum(y*np.log(prob1) + (1-y)*np.log(prob0))
    return log_likelihood

# Run trials
for j in range(num_trials):

    # Generate random component u ~ N(0, sigma)
    u = np.random.normal(0, sigma, num_obs)

    # Create x vector
    x = np.arange(1, num_obs+1)

    # Generate y_star and convert to y as a dummy variable
    y_star = alpha + beta * x + u
    y = (y_star > 300).astype(int)  # Changed from y_star > 300

    # Evaluate the log likelihood for each potential beta_hat
    max_log_likelihood = -np.inf
    beta_hat = 0
    for beta_tilda in np.arange(-1, 3.01, 0.01):
        print(log_likelihood(beta_tilda, alpha, x, y, sigma))
        current_log_likelihood = log_likelihood(beta_tilda, alpha, x, y, sigma)
        if current_log_likelihood > max_log_likelihood:
            max_log_likelihood = current_log_likelihood
            beta_hat = beta_tilda

    # Append the beta_hat value
    beta_hat_values.append(beta_hat)

# Convert list to a DataFrame
df = pd.DataFrame(beta_hat_values, columns=['beta_hat'])

# Plot histogram of beta_hat values
plt.hist(df['beta_hat'], bins=15, edgecolor='k')
plt.xlabel('Beta Hat')
plt.ylabel('Frequency')
plt.title('Histogram of Beta Hat')
plt.show()
