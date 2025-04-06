import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def f(x):
    x = np.array(x, ndmin=1)
    return ((6*x - 2)**2)*np.sin(12*x - 4)

def phi(z):
    return 1/np.sqrt(2*np.pi)*np.exp(-0.5*z**2)

def Phi(z):
    z = np.array(z, ndmin=1)
    return 0.5*(1 + tf.math.erf(z / np.sqrt(2))).numpy()

class GaussianProcess:
    def __init__(self):
        self.X = None
        self.Y = None
        self.R_inv = None
        self.beta = None
        self.sigma2 = None
        self.theta = None

    def corr_matrix(self, X1, X2, theta, jitter=1e-6):
        dist = (X1 - X2.T)**2
        R = np.exp(-theta * dist)
        if X1.shape[0] == X2.shape[0] and np.all(X1 == X2):
            R += jitter * np.eye(len(X1))
        return R

    def fit(self, X, Y):
        X = X.reshape(-1,1)
        Y = Y.reshape(-1,1)
        self.X = X
        self.Y = Y

        def neg_log_like(log_theta):
            theta_val = float(np.exp(log_theta))
            R = self.corr_matrix(self.X, self.X, theta_val)
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

        R = self.corr_matrix(self.X, self.X, self.theta)
        one = np.ones((len(X),1))
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R)
        beta = (one.T @ R_inv @ Y) / (one.T @ R_inv @ one)
        self.beta = float(beta)
        residual = Y - one * beta
        sigma2 = (residual.T @ R_inv @ residual) / len(X)

        self.R_inv = R_inv
        self.sigma2 = sigma2

    def predict(self, X_star):
        X_star = X_star.reshape(-1,1)
        r = self.corr_matrix(X_star, self.X, self.theta)
        y_pred = self.beta + (r @ self.R_inv @ (self.Y - self.beta)).ravel()
        s_list = []
        one = np.ones((len(self.X),1))
        for i in range(len(X_star)):
            ri = r[i:i+1, :]
            A = 1 - ri @ self.R_inv @ ri.T
            B = (1 - (one.T @ self.R_inv @ ri.T))**2 / (one.T @ self.R_inv @ one)
            s2 = self.sigma2 * (A + B)
            s_list.append(float(s2))
        s_list = np.sqrt(np.abs(s_list))
        return y_pred, s_list

def expected_improvement(x, f_min, gp):
    X_ = np.array([x]).reshape(-1,1)
    y_pred, s = gp.predict(X_)
    y_pred = y_pred[0]
    s = s[0]
    if s < 1e-12:
        return 0.0
    Z = (f_min - y_pred)/s
    ei = (f_min - y_pred)*Phi(Z) + s*phi(Z)
    return ei

def maximize_ei(gp, f_min, bounds=[0,1]):
    def neg_ei(x):
        return -expected_improvement(x, f_min, gp)
    best_ei = None
    best_x = None
    for start in np.linspace(bounds[0], bounds[1], 5):
        res = minimize(neg_ei, start, bounds=[bounds], method='L-BFGS-B')
        if best_ei is None or res.fun < best_ei:
            best_ei = res.fun
            best_x = res.x[0]
    return best_x

class DNNModel:
    def __init__(self, num_hidden_layers=3, neurons_per_layer=50, epochs=3000, lr=0.001):
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def build_model(self):
        inputs = tf.keras.Input(shape=(1,))
        x = inputs
        for _ in range(self.num_hidden_layers):
            x = tf.keras.layers.Dense(self.neurons_per_layer, activation='tanh')(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model

    def fit(self, X, Y):
        X = X.reshape(-1,1)
        Y = Y.reshape(-1,1)
        self.model = self.build_model()
        self.model.fit(X, Y, epochs=self.epochs, verbose=0)

    def predict(self, X):
        X = X.reshape(-1,1)
        return self.model.predict(X, verbose=0).ravel()

def egonn_expected_improvement(x, f_min, pred_model, uncert_model):
    X_ = np.array([x]).reshape(-1,1)
    y_pred = pred_model.predict(X_)
    u_pred = uncert_model.predict(X_)
    s = np.sqrt(np.abs(u_pred[0]))
    if s < 1e-12:
        return 0.0
    Z = (f_min - y_pred[0])/s
    ei = (f_min - y_pred[0])*Phi(Z) + s*phi(Z)
    return ei

def maximize_egonn_ei(pred_model, uncert_model, f_min, bounds=[0,1]):
    def neg_ei(x):
        return -egonn_expected_improvement(x, f_min, pred_model, uncert_model)
    best_ei = None
    best_x = None
    for start in np.linspace(bounds[0], bounds[1], 5):
        res = minimize(neg_ei, start, bounds=[bounds], method='L-BFGS-B')
        if best_ei is None or res.fun < best_ei:
            best_ei = res.fun
            best_x = res.x[0]
    return best_x

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    X_true = np.linspace(0,1,200)
    Y_true = f(X_true)
    X_ego = np.linspace(0,1,4)[:,None]
    Y_ego = f(X_ego)
    gp = GaussianProcess()
    max_iter = 8
    f_best_history_ego = []
    x_best_history_ego = []

    for i in range(max_iter):
        gp.fit(X_ego, Y_ego)
        x_best = X_ego[np.argmin(Y_ego)]
        f_min = np.min(Y_ego)
        f_best_history_ego.append(f_min)
        x_best_history_ego.append(x_best)
        next_x = maximize_ei(gp, f_min)
        next_y = f(next_x)[0]
        X_ego = np.vstack([X_ego, [[next_x]]])
        Y_ego = np.vstack([Y_ego, [[next_y]]])

    x_best_ego = X_ego[np.argmin(Y_ego)]
    f_best_ego = np.min(Y_ego)
    X_egonn = np.linspace(0,1,4)[:,None]
    Y_egonn = f(X_egonn)
    X_uncert = np.linspace(0,1,10)[:,None]
    Y_uncert_true = f(X_uncert)
    f_best_history_egonn = []
    x_best_history_egonn = []

    for i in range(max_iter):
        pred_model = DNNModel()
        pred_model.fit(X_egonn, Y_egonn)
        Y_uncert_pred = pred_model.predict(X_uncert)
        U2 = (Y_uncert_true.flatten() - Y_uncert_pred) ** 2
        U2 = U2.reshape(-1, 1)
        uncert_model = DNNModel()
        uncert_model.fit(X_uncert, U2)
        f_min = np.min(Y_egonn)
        x_best = X_egonn[np.argmin(Y_egonn)]
        f_best_history_egonn.append(f_min)
        x_best_history_egonn.append(x_best)
        next_x_egonn = maximize_egonn_ei(pred_model, uncert_model, f_min)
        next_y_egonn = f(next_x_egonn)[0]
        X_egonn = np.vstack([X_egonn, [[next_x_egonn]]])
        Y_egonn = np.vstack([Y_egonn, [[next_y_egonn]]])

    x_best_egonn = X_egonn[np.argmin(Y_egonn)]
    f_best_egonn = np.min(Y_egonn)
    gp.fit(X_ego, Y_ego)
    Y_gp_pred, Y_gp_std = gp.predict(X_true[:, None])

    plt.figure(figsize=(10, 6))
    plt.fill_between(X_true, Y_gp_pred - 2*np.array(Y_gp_std), Y_gp_pred + 2*np.array(Y_gp_std), color='r', alpha=0.2, label='GPR ±2σ')
    plt.plot(X_true, Y_true, 'k-', label='True function')
    plt.plot(X_true, Y_gp_pred, 'r-', label='GPR prediction')
    plt.plot(X_ego, Y_ego, 'bo', label='EGO samples')
    plt.axvline(x_best_ego, color='g', linestyle='--', label='EGO infill')
    plt.xlabel('x')
    plt.ylabel('$y(x)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("final_surrogate_ego_y_hat.eps", format='eps')
    plt.show()

    pred_model = DNNModel()
    pred_model.fit(X_egonn, Y_egonn)
    Y_dnn_pred = pred_model.predict(X_true[:, None])
    Y_uncert_pred = pred_model.predict(X_uncert)
    U2 = (Y_uncert_true.flatten() - Y_uncert_pred) ** 2
    U2 = U2.reshape(-1, 1)
    uncert_model = DNNModel()
    uncert_model.fit(X_uncert, U2)
    Y_uncert_fit = uncert_model.predict(X_true[:, None])
    Y_uncert_std = np.sqrt(np.abs(Y_uncert_fit)).ravel()

    plt.figure(figsize=(10, 6))
    plt.fill_between(X_true, Y_dnn_pred - 2*Y_uncert_std, Y_dnn_pred + 2*Y_uncert_std, color='r', alpha=0.2, label='DNN ±2σ')
    plt.plot(X_true, Y_true, 'k-', label='True function')
    plt.plot(X_true, Y_dnn_pred, 'r-', label='NN prediction')
    plt.plot(X_egonn, Y_egonn, 'bo', label='EGONN samples')
    plt.axvline(x_best_egonn, color='g', linestyle='--', label='EGONN infill')
    plt.xlabel('x')
    plt.ylabel('$y(x)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("final_surrogate_egonn_y_hat.eps", format='eps')
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(f_best_history_ego, 'r--', label='EGO best f')
    plt.plot(f_best_history_egonn, 'b-', label='EGONN best f')
    plt.xlabel('Iteration')
    plt.ylabel('$y_{\\min}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_ymin.eps", format='eps')
    plt.close()

    for i in range(max_iter):
        gp.fit(X_ego[:4+i, :], Y_ego[:4+i, :])
        Y_gp_pred, Y_gp_std = gp.predict(X_true[:, None])
        pred_model = DNNModel()
        pred_model.fit(X_egonn[:4+i, :], Y_egonn[:4+i, :])
        Y_dnn_pred = pred_model.predict(X_true[:, None])
        Y_uncert_pred = pred_model.predict(X_uncert)
        U2 = (Y_uncert_true.flatten() - Y_uncert_pred)**2
        U2 = U2.reshape(-1, 1)
        uncert_model = DNNModel()
        uncert_model.fit(X_uncert, U2)
        Y_uncert_fit = uncert_model.predict(X_true[:, None])
        Y_uncert_std = np.sqrt(np.abs(Y_uncert_fit)).ravel()
        f_min_ego = np.min(Y_ego[:4+i])
        EI_ego = np.array([expected_improvement(x, f_min_ego, gp) for x in X_true])
        f_min_egonn = np.min(Y_egonn[:4+i])
        EI_egonn = np.array([egonn_expected_improvement(x, f_min_egonn, pred_model, uncert_model) for x in X_true])
        x_infill_ego = X_ego[3+i]
        y_infill_ego = f(x_infill_ego)
        x_infill_egonn = X_egonn[3+i]
        y_infill_egonn = f(x_infill_egonn)

        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.plot(X_true, Y_true, 'k-', label='True function')
        plt.plot(X_true, Y_dnn_pred, 'b-', label='NN prediction')
        plt.plot(X_true, Y_gp_pred, 'g-', label='GPR prediction')
        plt.plot(X_egonn[:4+i], Y_egonn[:4+i], 'o', color='orange', label='EGONN samples')
        plt.plot(X_ego[:4+i], Y_ego[:4+i], 'o', color='darkblue', label='EGO samples')
        plt.plot([x_infill_egonn], [y_infill_egonn], 'g*', label='EGONN infill')
        plt.plot([x_infill_ego], [y_infill_ego], 'r*', label='EGO infill')
        plt.xlabel('x')
        plt.ylabel('$y(x)$')
        plt.legend()
        plt.grid(False)

        plt.subplot(1, 3, 2)
        plt.plot(X_true, Y_uncert_std, 'b-', label='NN $s(x)$')
        plt.plot(X_true, Y_gp_std, 'g-', label='GPR $s(x)$')
        # plt.plot([x_infill_egonn], [y_infill_egonn], 'g*', label='EGONN infill')
        # plt.plot([x_infill_ego], [y_infill_ego], 'r*', label='EGO infill')
        plt.xlabel('x')
        plt.ylabel('$s(x)$')
        plt.legend()
        plt.grid(False)

        plt.subplot(1, 3, 3)
        plt.plot(X_true, EI_egonn, 'b-', label='EGONN $EI(x)$')
        plt.plot(X_true, EI_ego, 'g-', label='EGO $EI(x)$')
        # plt.plot([x_infill_egonn], [y_infill_egonn], 'g*', label='EGONN infill')
        # plt.plot([x_infill_ego], [y_infill_ego], 'r*', label='EGO infill')
        plt.xlabel('x')
        plt.ylabel('$EI(x)$')
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f"iteration_{i+1}.eps", format='eps')
        plt.close()

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(range(1, max_iter+1), f_best_history_egonn, 'b-', label='EGONN')
    plt.plot(range(1, max_iter+1), f_best_history_ego, 'r-', label='EGO')
    plt.axhline(y=np.min(Y_true), color='k', linestyle='--', label='Global optimum')
    plt.xlabel('Iterations')
    plt.ylabel('$y_{\\min}$')
    plt.legend()
    plt.grid(False)

    plt.subplot(3, 1, 2)
    plt.plot(range(1, max_iter+1), [x[0] for x in x_best_history_egonn], 'b-', label='EGONN')
    plt.plot(range(1, max_iter+1), [x[0] for x in x_best_history_ego], 'r-', label='EGO')
    optimal_x = X_true[np.argmin(Y_true)]
    plt.axhline(y=optimal_x, color='k', linestyle='--', label='Global optimum')
    plt.xlabel('Iterations')
    plt.ylabel('$x^*$')
    plt.legend()
    plt.grid(False)

    plt.subplot(3, 1, 3)
    EImax_ego = [max(np.array([expected_improvement(x, np.min(Y_ego[:4+j]), gp) for x in X_true])) for j in range(max_iter)]
    EImax_egonn = [max(np.array([egonn_expected_improvement(x, np.min(Y_egonn[:4+j]), pred_model, uncert_model) for x in X_true])) for j in range(max_iter)]
    plt.plot(range(1, max_iter+1), EImax_egonn, 'b-', label='EGONN')
    plt.plot(range(1, max_iter+1), EImax_ego, 'r-', label='EGO')
    plt.xlabel('Iterations')
    plt.ylabel('$EI_{\\max}$')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("optimization_history.eps", format='eps')
    plt.close()