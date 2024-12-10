import numpy as np
from scipy.optimize import minimize

# Define parameters
R = 1000  # Number of Monte Carlo repetitions
n = 500  # Sample size
mu_0 = 4  # True mean
sigma2_0 = 4  # True variance

# Initialize arrays to store results
mu1_estimates = np.zeros(R)
sigma2_1_estimates = np.zeros(R)
mu2_estimates = np.zeros(R)
sigma2_2_estimates = np.zeros(R)


# Define the GMM criterion function for the second estimator
def gmm_objective(params, data):
    mu, sigma2 = params
    if sigma2 <= 0:  # Ensure variance is positive
        return np.inf
    g = np.array([
        mu - data,  # First moment condition (mean)
        sigma2 - (data - mu) ** 2,  # Second moment condition (variance)
        data ** 3 - mu * (mu ** 2 + 3 * sigma2)  # Third moment condition (skewness)
    ])
    moment_avg = np.mean(g, axis=1)
    return np.sum(moment_avg ** 2)  # Minimize squared moment conditions


# Monte Carlo simulation
for r in range(R):
    print(f"Repetition {r + 1}/{R}")
    # Generate a sample of size n from N(mu_0, sigma2_0)
    sample = np.random.normal(mu_0, np.sqrt(sigma2_0), n)

    # Estimator 1: Sample mean and variance (MLE)
    mu1_estimates[r] = np.mean(sample)
    sigma2_1_estimates[r] = np.var(sample, ddof=1)  # Unbiased sample variance

    # Estimator 2: GMM
    # Initial guesses for mu and sigma^2
    initial_guess = [np.mean(sample), np.var(sample)]
    # Numerical optimization for GMM
    result = minimize(
        gmm_objective, initial_guess, args=(sample,),
        bounds=[(None, None), (1e-6, None)]  # Ensure sigma^2 > 0
    )
    if result.success:
        mu2_estimates[r], sigma2_2_estimates[r] = result.x
    else:
        mu2_estimates[r], sigma2_2_estimates[r] = np.nan, np.nan  # Handle optimization failure

# Calculate MSE for both estimators
mse_mu1 = np.mean((mu1_estimates - mu_0) ** 2)
mse_sigma2_1 = np.mean((sigma2_1_estimates - sigma2_0) ** 2)
mse_mu2 = np.nanmean((mu2_estimates - mu_0) ** 2)  # Use nanmean to handle potential NaNs
mse_sigma2_2 = np.nanmean((sigma2_2_estimates - sigma2_0) ** 2)

# Print results
print("\nMSE for Estimator 1 (Sample mean and variance):")
print(f"  MSE(mu1) = {mse_mu1:.4f}")
print(f"  MSE(sigma2_1) = {mse_sigma2_1:.4f}")

print("\nMSE for Estimator 2 (GMM):")
print(f"  MSE(mu2) = {mse_mu2:.4f}")
print(f"  MSE(sigma2_2) = {mse_sigma2_2:.4f}")
