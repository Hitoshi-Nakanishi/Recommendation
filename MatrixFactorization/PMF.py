import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, coo_matrix
from sklearn.base import BaseEstimator

class PMF(BaseEstimator):

    def __init__(self, dims, dim_D, sigma2, lambda_, epoch_num=20, logger=None):
        self.dims = dims
        self.sigma2 = sigma2
        self.lambda_ = lambda_
        self.dim_D = dim_D
        self.epoch_num = epoch_num
        self.logger = logger

    def get_params(self, deep=True):
        return {'dims': self.dims, 'dim_D': self.dim_D, 'lambda_': self.lambda_, 'sigma2': self.sigma2}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X):
        """
        This method estimates MAP (Maximum a posteriori)
        X has three columns: (x,y)-cordinates, and z values
        """
        R = coo_matrix((X[:, 2], (X[:, 0], X[:, 1])))
        R = lil_matrix(R)

        self.U = np.zeros((self.epoch_num, self.dims['N'], self.dim_D))
        self.V = np.zeros((self.epoch_num, self.dim_D, self.dims['M']))
        self.U[0, :, :] = np.random.rand(self.dims['N'], self.dim_D)
        self.V[0, :, :] = np.random.rand(self.dim_D, self.dims['M'])
        self.sigma2I = np.identity(self.dim_D) * self.sigma2

        for t in range(self.epoch_num - 1):
            if self.logger is not None:
                self.logger.info(f'epoch: {t} 🤔 start to compute U')
            self.compute_U(t, self.U, self.V, R, self.sigma2I, self.lambda_, self.logger)
            if self.logger is not None:
                self.logger.info(f'epoch: {t} 😁 start to compute V')
            self.compute_V(t, self.V, self.U, R, self.sigma2I, self.lambda_, self.logger)
        return self

    @staticmethod
    def compute_U(t, U, V, R, sigma2I, lambda_, logger=None):
        N_dim = U.shape[1]
        for i in range(N_dim):
            row_idx = R.getrow(i).nonzero()[1]
            if len(row_idx) < 1:
                continue
            Rij_vals = R.getrow(i).data[0]
            v_vT = np.sum([np.outer(V[t, :, j], V[t, :, j].T) for j in row_idx], axis=0)
            inv_term = np.linalg.inv(lambda_ * sigma2I + v_vT)
            Rv = np.sum([Rij * V[t, :, j] for j, Rij in zip(row_idx, Rij_vals)], axis=0)
            U[t + 1, i, :] = inv_term.dot(Rv)
            if logger is not None and i> 0 and i%1000 == 0:
                logger.info(f'U: user {i} finished')

    @staticmethod
    def compute_V(t, V, U, R, sigma2I, lambda_, logger=None):
        M_dim = V.shape[2]
        for j in range(M_dim):
            col_idx = R.getcol(j).nonzero()[0]
            if len(col_idx) < 1:
                continue
            Rij_vals = R.getcol(j).data
            u_uT = np.sum([np.outer(U[t + 1, i, :], U[t + 1, i, :].T) for i in col_idx], axis=0)
            inv_term = np.linalg.inv(lambda_ * sigma2I + u_uT)
            Ru = np.sum([Rij * U[t + 1, i, :] for i, Rij in zip(col_idx, Rij_vals)], axis=0)
            V[t + 1, :, j] = inv_term.dot(Ru)
            if logger is not None and j > 0 and j%50000 == 0:
                logger.info(f'V: movie {j} finished')

    def score(self, X):
        """
        return MSE score based on last epoch's result
        """
        R = coo_matrix((X[:, 2], (X[:, 0], X[:, 1])))
        R = lil_matrix(R)
        t = self.epoch_num - 1
        val = self.compute_mse(R, self.U[t, :, :], self.V[t, :, :])
        return val

    @staticmethod
    def compute_mse(R, u, v):
        """
        compute MSE score using sparse lil_martix
        """
        e, m = 0., 0.
        for i in range(R.shape[0]):
            for j in R.getrow(i).nonzero()[1]:
                e += (R[i,j] - np.dot(u[i, :], v[:, j]))**2
                m += 1
        return e / m

    def compute_hitstorical_errors(self, X):
        R = coo_matrix((X[:, 2], (X[:, 0], X[:, 1])))
        R = lil_matrix(R)
        mse = {}
        for t in range(self.epoch_num):
            mse[t] = self.compute_mse(R, self.U[t, :, :], self.V[t, :, :])
        return pd.Series(mse)

    def predict_R(self, t):
        """
        this method is used for testing R prediction if R is small matrix
        """
        pred_R = self.U[t, :, :].dot(self.V[t, :, :])
        return pred_R

    @staticmethod
    def get_target_R(X):
        """
        this method is used for testing compute_mse method
        in small matrix, we tested
        ```
        t = 10
        target_R = pmf.get_target_R()
        pred_R = pmf.predict_R(t)
        mse_A = np.nanmean(np.square(pred_R - target_R))
        mse_B = pmf.compute_test_error(t)
        ```
        mse_A and mse_B should have same number
        """
        R = coo_matrix((X[:, 2], (X[:, 0], X[:, 1])))
        R = lil_matrix(R)
        target_R = R.toarray().astype(np.float32)
        target_R[target_R == 0] = np.nan
        return target_R