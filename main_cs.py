import argparse
import numpy as np
from simulate_data import sparse_linear_model as slm
from greedy_cs import orthogonal_matching_pursuit as omp

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, help="Sparsity level", default=2)
parser.add_argument('--M', type=int, help="Number of Parameters", default=10)
parser.add_argument('--N', type=int, help="Number of Data Points", default=100)
parser.add_argument('--SNR', type=float, help="Signal-to-Noise Ratio (db)",
                    default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    the_slm = slm(M=args.M, K=args.K)
    the_slm.generate_data(N=args.N, SNR=args.SNR)
    estimator = omp(the_slm.K)
    beta_hat = estimator.estimate(the_slm.X, the_slm.y)
    rmse = np.linalg.norm(the_slm.beta - beta_hat)
    print("The Root-Mean-Squarred-Error (RMSE) = {:6.3f}".format(rmse))
    correct_support = np.intersect1d(np.nonzero(the_slm.beta)[0],
                                     np.nonzero(beta_hat))
    print("The algorithm has identified {:d} nonzero element out of {:d} correctly.".
          format(len(correct_support), args.K))
    