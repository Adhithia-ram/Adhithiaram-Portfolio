import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def branin(x):
    x1 = x[...,0]
    x2 = x[...,1]
    a = 1.0
    b = 5.1/(4.0*np.pi**2)
    c = 5.0/np.pi
    r = 6.0
    s = 10.0
    t = 1.0/(8.0*np.pi)
    return (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s

def constraint_g(x):
    x1_star = 3.0 * np.pi
    x2_star = 2.475
    x1 = x[...,0]
    x2 = x[...,1]
    return 1.0 - (x1 * x2)/(x1_star * x2_star)

def penalty_H(x, lambda_=100.0):
    gx = constraint_g(x)
    penalty = (np.maximum(gx,0))
    return branin(x) + lambda_ * penalty

def phi(z):
    return 1/np.sqrt(2*np.pi)*np.exp(-0.5*z**2)

def Phi(z):
    z = np.array(z, ndmin=1)
    return 0.5*(1 + tf.math.erf(z / np.sqrt(2))).numpy()

def lhs(n, samples, seed=None):
    if seed is not None:
        np.random.seed(seed)
    cut = np.linspace(0,1, samples+1)
    u = np.random.rand(samples,n)
    a = cut[:samples]
    b = cut[1:]
    rdpoints = u*(b - a)[:,None] + a[:,None]
    result = np.zeros_like(rdpoints)
    for j in range(n):
        order = np.random.permutation(samples)
        result[:,j] = rdpoints[order,j]
    return result

class GaussianProcess:
    def __init__(self):
        self.X = None
        self.Y = None
        self.R_inv = None
        self.beta = None
        self.sigma2 = None
        self.theta = None

    def corr_matrix(self, X1, X2, theta, jitter=1e-10):
        dist = np.sum((X1[:,None,:] - X2[None,:,:])**2, axis=2)
        R = np.exp(-theta * dist)
        if X1.shape[0] == X2.shape[0] and np.allclose(X1, X2):
            R += jitter * np.eye(len(X1))
        return R

    def fit(self, X, Y):
        def neg_log_like(log_theta):
            theta_val = float(np.exp(log_theta))
            R = self.corr_matrix(X, X, theta_val)
            one = np.ones((len(X),1))
            try:
                R_inv = np.linalg.inv(R)
            except np.linalg.LinAlgError:
                return np.inf
            beta = (one.T @ R_inv @ Y) / (one.T @ R_inv @ one)
            residual = Y - one * beta
            sigma2 = (residual.T @ R_inv @ residual) / len(X)
            if sigma2 <= 0 or np.linalg.det(R) <= 0:
                return np.inf
            return float(len(X)*0.5*np.log(sigma2) + 0.5*np.log(np.linalg.det(R)))

        res = minimize(neg_log_like, np.log(1.0), method='L-BFGS-B')
        self.theta = float(np.exp(res.x))
        R = self.corr_matrix(X, X, self.theta)
        one = np.ones((len(X),1))
        R_inv = np.linalg.inv(R)
        beta = (one.T @ R_inv @ Y) / (one.T @ R_inv @ one)
        residual = Y - one * beta
        sigma2 = (residual.T @ R_inv @ residual) / len(X)

        self.X = X
        self.Y = Y
        self.R_inv = R_inv
        self.beta = float(beta)
        self.sigma2 = sigma2

    def predict(self, X_star):
        r = self.corr_matrix(X_star, self.X, self.theta)
        y_pred = self.beta + (r @ self.R_inv @ (self.Y - self.beta))
        s_list = []
        one = np.ones((len(self.X),1))
        for i in range(len(X_star)):
            ri = r[i:i+1, :]
            A = 1 - ri @ self.R_inv @ ri.T
            B = (1 - (one.T @ self.R_inv @ ri.T))**2/(one.T @ self.R_inv @ one)
            s2 = self.sigma2*(A + B)
            s_list.append(float(s2))
        s_list = np.sqrt(np.abs(s_list))
        return y_pred.ravel(), s_list

def expected_improvement(x, f_min, gp):
    X_ = x.reshape(1,-1)
    y_pred, s = gp.predict(X_)
    y_pred = y_pred[0]
    s = s[0]
    if s < 1e-12:
        return 0.0
    Z = (f_min - y_pred)/s
    ei = (f_min - y_pred)*Phi(Z) + s*phi(Z)
    return ei

def maximize_ei(gp, f_min, bounds):
    def neg_ei(X):
        return -expected_improvement(X, f_min, gp)
    best_val = None
    best_x = None
    for _ in range(5):
        start = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        res = minimize(neg_ei, start, bounds=bounds, method='L-BFGS-B')
        if best_val is None or res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    return best_x

class DNNModel:
    def __init__(self, input_dim=2, epochs=2000, lr=0.001):
        self.epochs = epochs
        self.lr = lr
        self.input_dim = input_dim
        self.model = None

    def build_model(self):
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = tf.keras.layers.Dense(8, activation='relu')(inputs)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        self.model = self.build_model()
        self.model.fit(X, Y, epochs=self.epochs, verbose=0)

    def predict(self, X):
        X = np.array(X)
        return self.model.predict(X, verbose=0).ravel()

def egonn_expected_improvement(x, f_min, pred_model, uncert_model):
    X_ = x.reshape(1,-1)
    y_pred = pred_model.predict(X_)
    u_pred = uncert_model.predict(X_)
    s = np.sqrt(np.abs(u_pred[0]))
    if s < 1e-12:
        return 0.0
    Z = (f_min - y_pred[0])/s
    ei = (f_min - y_pred[0])*Phi(Z) + s*phi(Z)
    return ei

def maximize_egonn_ei(pred_model, uncert_model, f_min, bounds):
    def neg_ei(X):
        return -egonn_expected_improvement(X, f_min, pred_model, uncert_model)
    best_val = None
    best_x = None
    for _ in range(5):
        start = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        res = minimize(neg_ei, start, bounds=bounds, method='L-BFGS-B')
        if best_val is None or res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    return best_x

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    x_opt = np.array([3*np.pi, 2.475])
    f_opt = branin(x_opt.reshape(1,-1))[0]

    bounds = [(-5,10), (0,15)]
    lambda_ = 100.0

    N_init = 30
    X_lhs = lhs(2, N_init, seed=42)
    X_init = np.zeros((N_init,2))
    for i,(low,high) in enumerate(bounds):
        X_init[:,i] = X_lhs[:,i]*(high - low) + low
    Y_init = penalty_H(X_init, lambda_=lambda_).reshape(-1,1)

    max_iter = 40
    N_uncert = 100
    X_uncert_lhs = lhs(2, N_uncert, seed=10)
    X_uncert = np.zeros((N_uncert,2))
    for i,(low,high) in enumerate(bounds):
        X_uncert[:,i] = X_uncert_lhs[:,i]*(high - low) + low

    X_ego = np.copy(X_init)
    Y_ego = np.copy(Y_init)
    X_egonn = np.copy(X_init)
    Y_egonn = np.copy(Y_init)

    f_best_history_ego = []
    x_best_history_ego = []
    ei_max_history_ego = []

    f_best_history_egonn = []
    x_best_history_egonn = []
    ei_max_history_egonn = []

    EI_sample_count = 200
    def generate_ei_samples():
        pts = np.zeros((EI_sample_count,2))
        for i,(low,high) in enumerate(bounds):
            pts[:,i] = np.random.uniform(low, high, EI_sample_count)
        return pts

    for i in range(max_iter):
        gp = GaussianProcess()
        gp.fit(X_ego, Y_ego)
        f_min_ego = np.min(Y_ego)
        x_min_ego = X_ego[np.argmin(Y_ego)]
        f_best_history_ego.append(f_min_ego)
        x_best_history_ego.append(x_min_ego)
        next_x_ego = maximize_ei(gp, f_min_ego, bounds)
        next_y_ego = penalty_H(next_x_ego.reshape(1,-1), lambda_=lambda_)[0]
        X_ego = np.vstack((X_ego, next_x_ego))
        Y_ego = np.vstack((Y_ego, next_y_ego))

        pred_model = DNNModel(input_dim=2)
        pred_model.fit(X_egonn, Y_egonn)
        Y_uncert_true = penalty_H(X_uncert, lambda_=lambda_)
        Y_uncert_pred = pred_model.predict(X_uncert)
        U2 = (Y_uncert_true - Y_uncert_pred)**2
        U2 = U2.reshape(-1,1)
        uncert_model = DNNModel(input_dim=2)
        uncert_model.fit(X_uncert, U2)
        f_min_egonn = np.min(Y_egonn)
        x_min_egonn = X_egonn[np.argmin(Y_egonn)]
        f_best_history_egonn.append(f_min_egonn)
        x_best_history_egonn.append(x_min_egonn)
        next_x_egonn = maximize_egonn_ei(pred_model, uncert_model, f_min_egonn, bounds)
        next_y_egonn = penalty_H(next_x_egonn.reshape(1,-1), lambda_=lambda_)[0]
        X_egonn = np.vstack((X_egonn, next_x_egonn))
        Y_egonn = np.vstack((Y_egonn, next_y_egonn))

        ei_points = generate_ei_samples()
        EIs_ego = [expected_improvement(pt, f_min_ego, gp) for pt in ei_points]
        ei_max_history_ego.append(np.max(EIs_ego))
        EIs_egonn = [egonn_expected_improvement(pt, f_min_egonn, pred_model, uncert_model) for pt in ei_points]
        ei_max_history_egonn.append(np.max(EIs_egonn))

    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    plt.plot(f_best_history_egonn, 'b-', label='EGONN')
    plt.plot(f_best_history_ego, 'r--', label='EGO')
    plt.axhline(y=f_opt, color='k', linestyle=':', label='Global optimum')
    plt.xlabel('Iterations')
    plt.ylabel('y_min')
    # plt.title('Minimum Observation')
    plt.tight_layout()
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot([x[0] for x in x_best_history_egonn], 'b-', label='EGONN')
    plt.plot([x[0] for x in x_best_history_ego], 'r--', label='EGO')
    plt.axhline(y=x_opt[0], color='k', linestyle=':', label='Global optimum')
    plt.xlabel('Iterations')
    plt.ylabel('x1*')
    # plt.title('X1 Location of Minimum Observation')
    plt.tight_layout()
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot([x[1] for x in x_best_history_egonn], 'b-', label='EGONN')
    plt.plot([x[1] for x in x_best_history_ego], 'r--', label='EGO')
    plt.axhline(y=x_opt[1], color='k', linestyle=':', label='Global optimum')
    plt.xlabel('Iterations')
    plt.ylabel('x2*')
    # plt.title('X2 Location of Minimum Observation')
    plt.tight_layout()
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(ei_max_history_egonn, 'b-', label='EGONN')
    plt.plot(ei_max_history_ego, 'r--', label='EGO')
    plt.xlabel('Iterations')
    plt.ylabel('EI_max')
    # plt.title('Maximum Expected Improvement')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimization_progress.eps', format='eps')
    plt.show()

    x1_lin = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2_lin = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1_grid, X2_grid = np.meshgrid(x1_lin, x2_lin)
    X_grid = np.stack([X1_grid, X2_grid], axis=-1)
    Z = branin(X_grid)
    G = constraint_g(X_grid)

    plt.figure(figsize=(8,6))
    CS = plt.contour(X1_grid, X2_grid, Z, levels=20, cmap='viridis')
    cbar = plt.colorbar(CS)
    cbar.set_label('Branin Value')
    plt.contour(X1_grid, X2_grid, G, levels=[0], colors='red', linestyles='--', linewidths=2)
    plt.plot(X_init[:,0], X_init[:,1], 'o', color='blue', label='Initial dataset')
    plt.plot(X_ego[N_init:,0], X_ego[N_init:,1], 'x', color='red', label='EGO infill')
    plt.plot(X_egonn[N_init:,0], X_egonn[N_init:,1], '*', color='black', label='EGONN infill')
    plt.plot(x_opt[0], x_opt[1], 'd', color='orange', label='Global Optimum')
    plt.xlabel('x1')
    plt.ylabel('x2')
    # plt.title('True Function Contours and Samples')
    plt.legend()
    plt.tight_layout()
    plt.savefig('branin_function_contour.eps', format='eps')
    plt.show()
