import numpy as np
from scipy.stats import multivariate_normal


def multivariate_gaussian_distribution(mu, Sigma, evaluation_locations):
    d = len(mu)
    det_sigma = np.linalg.det(Sigma)
    inv_sigma = np.linalg.inv(Sigma)

    Gaussian_constant = 1 / ((2 * np.pi) ** (d/2) * det_sigma ** 0.5)

    densities = []
    for x in evaluation_locations:
        diff = x - mu
        exponent = -0.5 * (diff.T @ inv_sigma @ diff)
        pdf = Gaussian_constant * np.exp(exponent)
        densities.append(pdf)

    return np.array(densities)

# Parameters
mu = np.array([1, 3, 5])
Sigma = np.array([[4, 2, 1],
                  [2, 5, 2],
                  [1, 2, 3]])
locations = np.array([[2, 2, 2],
                      [1, 4, 3],
                      [1, 1, 5]])

# Your implementation
my_results = multivariate_gaussian_distribution(mu, Sigma, locations)

print("My results:   ", my_results)
