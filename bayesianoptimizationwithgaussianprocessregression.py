import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import math

def normalfunc(x, d, sig=1, mu=0.5):
    rawterm1 = 1 / (sig * (2 * np.pi) ** 2)
    rawterm2 = -0.5 * ((x - mu) / sig) ** 2
    rawNorm = rawterm1 * np.exp(rawterm2)
    maxterm2 = 0
    maxNorm = rawterm1 * np.exp(maxterm2)
    minterm2 = -0.5 * ((0 - mu) / sig) ** 2
    minNorm = rawterm1 * np.exp(minterm2)
    return ((rawNorm - minNorm) / (maxNorm - minNorm)) / d

def normalsample(X, noise_level=0.05):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    Y = []
    for ii in range(n):
        y = 0
        for jj in range(d):
            y += normalfunc(X[ii, jj], d)
       
        y += np.random.normal(0, noise_level)
        Y.append(y)
    return Y

def objective_function(x):
    return -normalsample(np.array([x]))[0]

def perform_gpr(X, y, noise_level=0.001):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Add a WhiteKernel for noise
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level, noise_level_bounds=(1e-5, 1e-3))
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, optimizer='fmin_l_bfgs_b', random_state=0)
    gp.fit(X_scaled, y)
    return gp, scaler

def acquisition_function(x, gp, y_opt, xi=0.01, scaler=None):
    x_scaled = scaler.transform(np.array([x]).reshape(-1, 1))
    mean, std = gp.predict(x_scaled, return_std=True)
    z = (mean - y_opt - xi) / std
    return (mean - y_opt - xi) * norm.cdf(z) + std * norm.pdf(z)

def optimize_with_gpr(X_init, y_init, n_iter=10, max_iter_factor=4):
    X = X_init
    y = y_init
    gp = None
    scaler = None  

    bounds = [(0, 1)]  

    for _ in range(n_iter):
        gp, scaler = perform_gpr(X, y)
        acquisition_func = lambda x, gp=gp, scaler=scaler: -acquisition_function(x, gp, np.max(y), scaler=scaler)
        result = minimize(acquisition_func, np.random.rand(1), method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000})
        x_next = np.array([result.x[0]])
        y_next = normalsample(x_next)
        
        X = np.vstack([X, x_next.reshape(1, -1)])
        y = np.append(y, y_next)

    return X, y, gp, scaler

max_ter = 10 * 2

plt.figure(figsize=(12, 6))

x_true = np.linspace(0, 1, 1000)[:, np.newaxis]
y_true = normalsample(x_true)
plt.plot(x_true, y_true, 'k', lw=1, label='True Function', zorder=9)

initial_samples = 5
X_init = np.random.rand(initial_samples, 1)
y_init = normalsample(X_init)

X_init = np.array([[0.1], [0.2], [0.3]])
y_init = normalsample(X_init)

optimized_X, optimized_y, gp, scaler = optimize_with_gpr(X_init, y_init, n_iter=max_ter)

x_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
x_plot_scaled = scaler.transform(x_plot)
y_mean, _ = gp.predict(x_plot_scaled, return_std=True)

# Modified code to include inverse scaling for x_plot
x_plot_original_scale = scaler.inverse_transform(x_plot_scaled)
y_mean_original_scale, _ = gp.predict(x_plot_scaled, return_std=True)

plt.plot(x_plot_original_scale, y_mean_original_scale, 'r', lw=1, label='Bayesian Optimization Result', zorder=9)
plt.scatter(optimized_X, optimized_y, c='r', s=50, label='Bayesian Optimization Samples', zorder=10, edgecolors=(0, 0, 0))

plt.title('Bayesian Optimization with Gaussian Process Regression')
plt.legend()
plt.show()

# Surrogate function with inverse scaling
def normalfunc_surrogate(x, d, sig=1, mu=0.5):
    rawterm1 = 1 / (sig * (2 * math.pi) ** 2)
    rawterm2 = -0.5 * ((x - mu) / sig) ** 2
    rawNorm = rawterm1 * math.exp(rawterm2)
    maxterm2 = 0
    maxNorm = rawterm1 * math.exp(maxterm2)
    minterm2 = -0.5 * ((0 - mu) / sig) ** 2
    minNorm = rawterm1 * math.exp(minterm2)
    y = ((rawNorm - minNorm) / (maxNorm - minNorm)) / d
    return y

def normalsample_surrogate_inverse_scaling(X, scaler):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, _ = X.shape
    Y = []
    for ii in range(n):
        x_scaled = scaler.transform(np.array([X[ii]]).reshape(-1, 1))
        y = 0
        for jj in range(X.shape[1]):
            y += normalfunc_surrogate(x_scaled[0, jj], 1)
        Y.append(y)
    return Y

X_surrogate = np.random.rand(5, 1)
y_surrogate = normalsample_surrogate_inverse_scaling(X_surrogate, scaler)

plt.figure(figsize=(12, 6))

x_surrogate_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
x_surrogate_plot_scaled = scaler.transform(x_surrogate_plot)
y_surrogate_plot = normalsample_surrogate_inverse_scaling(x_surrogate_plot_scaled, scaler)
plt.plot(x_surrogate_plot, y_surrogate_plot, 'b', lw=1, label='Surrogate Function')

plt.scatter(X_surrogate, y_surrogate, c='g', s=50, label='Surrogate Function Samples', edgecolors=(0, 0, 0))

plt.title('Surrogate Function')
plt.legend()
plt.show()
