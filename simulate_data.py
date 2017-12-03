import numpy as np

class linear_model:
    """
    The class for simulating a linear model
    """
    def __init__(self, M):
        """
        class constructor
        
        Params:
            M: the number of parameters
        Attributes:
            M: the number of parameters
            beta: the parameter vector
        """
        assert isinstance(M,int), "Number of parameters M should be an integer number"
        self.M = M
        self.beta = np.random.normal(loc=0.0, scale=1.0, size=[M])
                
    def generate_data (self, N, SNR=None):
        """
        randomly draws a regressor matrix X and generates N data points
        
        Params: 
            N: the data size
            SNR: the Signal-to-Noise Ratio (in db)
        """
        self.N = N
        self.SNR = SNR
        self.X = np.random.normal(loc=0.0, scale=1.0, size=[N,self.M])
        
        self.y = np.dot(self.X, self.beta)
        
        if SNR is not None:
            scale_noise = np.sqrt(self.M)*10**(-SNR/20)
            self.y = self.y + np.random.normal(loc=0.0, scale=scale_noise, size=[N])
        return self.y 
        

class sparse_linear_model (linear_model):
    def __init__(self, M, K):
        """
        Constructor
        N: Size of parameter vector
        K: Sparsity; number of nonzero elements of parameter vector
        Notice that M>>K
        """
        assert K<=M, "The sparsity level K should not be bigger than the number of parameters"
        linear_model.__init__(self, M)
        self.K = K
        zero_els = np.random.choice(M, size=M-K, replace=False)
        self.beta[zero_els] = 0
        
    def generate_data(self, N, SNR=None):
        if SNR is not None:
            self.nz_SNR = SNR-10*np.log10(self.K/self.M) #non-zero SNR
            linear_model.generate_data(self, N, SNR=self.nz_SNR)
            self.SNR = SNR
        else:
            linear_model.generate_data(self, N, SNR)
 