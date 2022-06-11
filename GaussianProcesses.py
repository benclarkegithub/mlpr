import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(x1, x2, sigma_f, l):
    return (sigma_f ** 2) * np.exp(-0.5 * ((x1 - x2) ** 2).sum() / (l ** 2))

N = 101
X = np.linspace(0, 10, N)
mu = np.zeros(N)
Sigma = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        Sigma[i, j] = rbf_kernel(X[i], X[j], 2, 1)

plt.title("Random draws from an RBF kernel")
# Draw M functions from RBF kernel
M = 3
for m in range(M):
    Y = np.random.multivariate_normal(mu, Sigma)
    plt.plot(X, Y)
plt.show()

# Posterior graph
fig, ax = plt.subplots()
fig.suptitle("Random draws from the Gaussian process posterior")

# Create a random function from 6 RBFs
def rbf(x, c, h):
    return np.exp(-((x - c) ** 2) / (h ** 2))

C = np.linspace(0, 10, 6)
Phi = rbf(X[:, None], C, 2)
w = np.random.randn(6)
true_values = Phi @ w
ax.plot(X, true_values, label="True function")

sigma_y = 0.25
observed_i = np.floor(np.random.rand(5) * N).astype(int)
observed_loc = X[observed_i]
observed_val = true_values[observed_i] + (np.random.randn(5) * sigma_y)
ax.plot(observed_loc, observed_val, 'x', label="Observed values")

# Make the matrices
A = np.zeros((N, N))
B = np.zeros((5, 5))
CC = np.zeros((N, 5))

for i in range(N):
    for j in range(N):
        A[i, j] = rbf_kernel(X[i], X[j], 1, 1)

for i in range(5):
    for j in range(5):
        B[i, j] = rbf_kernel(observed_loc[i], observed_loc[j], 1, 1)
B = B + (sigma_y * np.identity(5))

for i in range(N):
    for j in range(5):
        CC[i, j] = rbf_kernel(X[i], observed_loc[j], 1, 1)

# p(X | observed_loc) = N(X; a + CC B^-1(g - b), A - CC B^-1 CC^T)
mu_posterior = mu + (CC @ np.linalg.inv(B) @ observed_val)
Sigma_posterior = A - (CC @ np.linalg.inv(B) @ CC.T)

# Fill between 1 std
upper = mu_posterior + np.sqrt(Sigma_posterior.diagonal())
lower = mu_posterior - np.sqrt(Sigma_posterior.diagonal())
ax.fill_between(X, lower, upper, alpha=0.5)

# Draw MM functions from RBF kernel
MM = 3
for m in range(MM):
    Y = np.random.multivariate_normal(mu_posterior, Sigma_posterior)
    ax.plot(X, Y)
ax.legend()
plt.show()
