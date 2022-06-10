import numpy as np
import matplotlib.pyplot as plt

# Linear and quadratic priors
w_l_0 = np.array([0, 0])
w_q_0 = np.array([0, 0, 0])

sigma_w_sq = 3 ** 2
V_l_0 = sigma_w_sq * np.identity(2)
V_q_0 = sigma_w_sq * np.identity(3)

# Set plots up
N = 101
X = np.linspace(-5, 5, N)
design_linear = np.row_stack((np.ones(N), X))
design_quadratic = np.row_stack((np.ones(N), X, X**2))

# Get M random samples from each
def draw_random_weights(M, w_l, V_l, w_q, V_q):
    linear = np.random.multivariate_normal(w_l, V_l, M)
    quadratic = np.random.multivariate_normal(w_q, V_q, M)
    return linear, quadratic

M = 3
linear, quadratic = draw_random_weights(M, w_l_0, V_l_0, w_q_0, V_q_0)

def plot_Y(Y_linear, Y_quadratic):
    for m in range(Y_linear.shape[0]):
        plt.plot(X, Y_linear[m], 'r')
    for m in range(Y_quadratic.shape[0]):
        plt.plot(X, Y_quadratic[m], 'b')
    plt.show()

Y_linear = linear @ design_linear
Y_quadratic = quadratic @ design_quadratic
plt.title("Random draws from linear and quadratic prior")
plot_Y(Y_linear, Y_quadratic)

# Create some data that is based on a quadratic
w = np.random.randn(3)
target_function = w @ design_quadratic
L = 5
points = np.floor(np.random.rand(L) * N).astype(int)
sigma_y = 1.5
sigma_y_sq = sigma_y ** 2
# Get the points and add some noise
observed_points = target_function[points] + (np.random.randn(L) * sigma_y_sq)

plt.plot(X, target_function, 'g', label="Target function")
plt.plot(X[points], observed_points, 'x', markersize=10, label="Observed points")
plt.legend()

# Calculate the posterior
def posterior(w, V, Phi, y, sigma_y_sq):
    V_inv = np.linalg.inv(V)
    V_N = sigma_y_sq * np.linalg.inv((sigma_y_sq * V_inv) + (Phi.T @ Phi))
    w_N = (V_N @ V_inv @ w) + ((1 / sigma_y_sq) * (V_N @ Phi.T @ y))
    return w_N, V_N

w_l_L, V_l_L = posterior(w_l_0, V_l_0, design_linear[:, points].T, observed_points, sigma_y_sq)
w_q_L, V_q_L = posterior(w_q_0, V_q_0, design_quadratic[:, points].T, observed_points, sigma_y_sq)

M = 3
linear_p, quadratic_p = draw_random_weights(M, w_l_L, V_l_L, w_q_L, V_q_L)
Y_linear_p = linear_p @ design_linear
Y_quadratic_p = quadratic_p @ design_quadratic
plt.title("Random draws from linear and quadratic posterior")
plot_Y(Y_linear_p, Y_quadratic_p)

# Calculate p(y | X, M) = integral p(y, w | X, M) dw = integral p(y | X, w, M) * p(w | M) dw
# Create sets of weights to integrate over
limit = 10
segments = 50
linspace_segments = np.linspace(-limit, limit, segments)
w_linear = []
w_quadratic = []
for w1_i, w1 in enumerate(linspace_segments):
    for w2_i, w2 in enumerate(linspace_segments):
        w_linear.append((w1, w2))
        for w3_i, w3 in enumerate(linspace_segments):
            w_quadratic.append((w1, w2, w3))

def normal(x, mu, sigma_sq):
    return (1 / np.sqrt(2 * np.pi * sigma_sq)) * np.exp(-0.5 * ((x - mu) ** 2) / sigma_sq)

# p(y | X, w, M)
p_y_X_w_M_linear = []
p_y_X_w_M_quadratic = []
# p(w | M)
p_w_linear = []
p_w_quadratic = []

for w1, w2 in w_linear:
    # p(y | X, w, M)
    predicted_points = np.array([w1, w2]) @ design_linear[:, points]
    p = normal(predicted_points, observed_points, sigma_y).prod()
    p_y_X_w_M_linear.append(p)

    # p(w | M)
    p_w1 = normal(w1, w_l_0[0], V_l_0[0, 0])
    p_w2 = normal(w2, w_l_0[1], V_l_0[1, 1])
    # Posterior
    # p_w1 = normal(w1, w_l_L[0], V_l_L[0, 0])
    # p_w2 = normal(w2, w_l_L[1], V_l_L[1, 1])
    p_w_linear.append(p_w1 * p_w2)

for w1, w2, w3 in w_quadratic:
    # p(y | X, w, M)
    predicted_points = np.array([w1, w2, w3]) @ design_quadratic[:, points]
    p = normal(predicted_points, observed_points, sigma_y).prod()
    p_y_X_w_M_quadratic.append(p)

    # p(w | M)
    p_w1 = normal(w1, w_q_0[0], V_q_0[0, 0])
    p_w2 = normal(w2, w_q_0[1], V_q_0[1, 1])
    p_w3 = normal(w3, w_q_0[2], V_q_0[2, 2])
    # Posterior
    # p_w1 = normal(w1, w_q_L[0], V_q_L[0, 0])
    # p_w2 = normal(w2, w_q_L[1], V_q_L[1, 1])
    # p_w3 = normal(w3, w_q_L[2], V_q_L[2, 2])
    p_w_quadratic.append(p_w1 * p_w2 * p_w3)

p_y_X_w_M_linear = np.array(p_y_X_w_M_linear)
p_y_X_w_M_quadratic = np.array(p_y_X_w_M_quadratic)
p_w_linear = np.array(p_w_linear)
p_w_quadratic = np.array(p_w_quadratic)

# p(y | X, M)
linear_delta = (2 * limit) / (segments ** 2)
quadratic_delta = (2 * limit) / (segments ** 3)

# Approximate the integral by summing all of the values together, and multiplying
# the result by delta.
p_y_X_linear = (p_y_X_w_M_linear * p_w_linear).sum() * linear_delta
p_y_X_quadratic = (p_y_X_w_M_quadratic * p_w_quadratic).sum() * quadratic_delta

print(f"p(y | X, M='Linear') = {p_y_X_linear}")
print(f"p(y | X, M='Quadratic') = {p_y_X_quadratic}")
