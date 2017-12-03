import numpy as np

class orthogonal_matching_pursuit:
    def __init__(self, K):
        """
        Constructor
        
        Params:
            K: the sparsity
        """
        self.K = K
        
    def estimate(self, X, y):
        """
        estimates the K-sparse oarameter vector using OMP method
        
        Params:
            y: the data vector
            X: the regressor matrix
            
        Returns:
            beta_hat: the K-sparse parameter vector
        """
        self.y = y
        self.X = X
        M = np.shape(X)[1]
        beta_hat = np.zeros(shape=[M])
        nz_elements = []
        for m in range(self.K):
            res = self.y - np.dot(self.X, beta_hat)
            corr_coeffs = np.dot(res, self.X)
            kmax = np.argmax(abs(corr_coeffs))
            nz_elements.append(kmax)
            beta_nz = np.linalg.lstsq(a=self.X[:,nz_elements], b=self.y)[0]
            beta_hat[nz_elements] = beta_nz
        return beta_hat
            
        
            