import shutil
import unittest
from numpy.linalg import inv
from scipy.spatial.distance import squareform, pdist
from latnet.latnet import Latnet
from latnet.expr_logger import Logger
import numpy as np


class TestLATNET(unittest.TestCase):
    """
    This class checks whether the output of LATNET is correct for a sample data. The output is saved in a file
    and then for testing, the current output of the program is compared with the one saved in the file.
    """

    @staticmethod
    def create_true_outputs():
        """
        Runs LATNET and saves output in '../test/latnet.tst'.
        """

        Logger.init_logger(None, 'LATNET')
        log_capture_string = Logger.get_string_logger()

        TestLATNET.run_latnet()

        with open('../test/latnet.tst', 'w') as fd:
            log_capture_string.seek(0)
            shutil.copyfileobj(log_capture_string, fd)

    @staticmethod
    def run_latnet():
        """
        Generates sample data, runs LATNET and saves output in a file.
        """

        np.random.seed(1200)
        N = 10
        T = 220
        sigma2_n = .2
        sigma2_g = 0.0001
        V = np.zeros((N, N))
        while (abs(V).sum() == 0):
            V = np.zeros((N, N))
            A = np.random.randint(2, size=(N * (N + 1)) / 2)
            W = np.random.normal(0, np.sqrt(1. / (0.5 * N)), size=(N * (N + 1)) / 2)
            V[np.tril_indices(N)] = A * W
            V[np.diag_indices(N)] = 0
        V = np.zeros(V.shape)
        V[1, 0] = .5
        V[2, 0] = .5
        Y, t, Kz = TestLATNET.generate_samples_synch(V, T, sigma2_n=sigma2_n, sigma2_g=sigma2_g)
        norm_t = (t[0] - np.mean(t[0])) / np.double(np.std(t[0]))

        import tensorflow as tf
        tf.set_random_seed(1010)
        elbo_, sigma2_n_, sigma2_g_, mu, sigma2_, alpha_, lengthscale, variance = Latnet.optimize(norm_t,
                                                                                                  np.hstack(Y),
                                                                                                  ['var', 'hyp'],
                                                                                                  2,
                                                                                                  {'var': 5, 'hyp': 5},
                                                                                                  Logger.logger,
                                                                                                  init_sigma2_n=sigma2_n,
                                                                                                  init_sigma2_g=sigma2_g,
                                                                                                  init_variance=2.,
                                                                                                  init_lengthscle=0.01,
                                                                                                  lambda_prior=0.5,
                                                                                                  lambda_postetior=2. / 3.,
                                                                                                  log_every=2,
                                                                                                  seed=0
                                                                                                  )
        Logger.logger.debug('alpha:' + str(alpha_))
        Logger.logger.debug('mu:' + str(mu))
        Logger.logger.debug('sigma2:' + str(sigma2_))
        Logger.logger.debug('hyp:' + str(np.array([sigma2_n_, sigma2_g_, lengthscale, variance])))

        # time can be different on each run - so not logging for time
        # Logger.logger.debug('time:' + str([start_time, end_time, end_time - start_time]))

    def test_latnet(self):
        """
        Runs LATNET and compares the log with the output saved in '../test/latnet.tst'.
        """

        Logger.init_logger(None, 'LATNET')
        # Logger.remove_handlers()
        with open('../test/latnet.tst', 'r') as myfile:
            true_output = myfile.read()

        test_output_buf = Logger.get_string_logger()

        TestLATNET.run_latnet()

        test_output_buf.seek(0)
        test_output = test_output_buf.read()

        self.assertEqual(true_output, test_output)


    @staticmethod
    def generate_samples_synch(V, n_samples=1000, sigma2_n=1., sigma2_g=1., variance=2., lengthscale=.01):
        """
        Generates synthesis samples from the model

        Args:

            V: numpy array of size N x N containing connectivity from j to i.
            n_samples: number of samples, i.e., observations per node.
            sigma2_n: variance of observation noise.
            sigma2_g: variance of connection noise.
            variance: variance of kernel.
            lengthscale: length-scale of kernel.

        Returns:
            : array of size N; each element contains a numpy array of size n_samples x 1.
            : array of size 1 with one element of size n_samples x 1 containing times of observations.
            : numpy array of size N x N containing kernel values between observation times.

        """

        Logger.logger.debug("started generating samples")

        V = V.T
        N = V.shape[0]
        # drawing samples from z_j
        ti = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, 1))

        Kz = TestLATNET.kernel(ti, variance, lengthscale)

        Z = np.zeros((N, n_samples))
        for n in range(N):
            Z[n, :] = np.reshape(np.random.multivariate_normal(np.zeros(ti.shape[0]), Kz), (ti.shape[0]))

        samples_ng = np.dot(V, np.random.normal(0, np.sqrt(sigma2_g), ti.shape[0] * N).reshape(N, ti.shape[0]))
        right_size = Z + samples_ng
        f = np.dot(inv(np.eye(N) - V), right_size)

        samples_nz = np.random.normal(0, np.sqrt(sigma2_n), ti.shape[0] * N).reshape(N, ti.shape[0])
        Y = f + samples_nz
        Y_array = []
        for n in range(N):
            Y_array.append(Y[n, :, np.newaxis])
        Logger.logger.debug("finished generating samples")
        return Y_array, [ti], Kz

    @staticmethod
    def kernel(X, variance, lengthscale):
        """
        RBF kernel in python.

        Args:
            X: numpy array of size T x 1 containing location of points.
            variance: double. variance of the kernel.
            lengthscale: double. length-scale of the kernel.


        Returns:
            : numpy array of size T x T containing kernel between `X' values.

        """

        pairwise_dists = squareform(pdist(X, 'euclidean'))
        return np.exp(-(pairwise_dists ** 2) / 2.0 / lengthscale ** 2) * variance + \
                    1e-5 * np.identity(X.shape[0])


if __name__ == '__main__':

    # use this for generating and saving output in '../test/latnet.tst' for checking against the output in the future.
    # TestLATNET.create_true_outputs()

    suite = unittest.TestLoader().loadTestsFromTestCase(TestLATNET)
    unittest.TextTestRunner(verbosity=2).run(suite)
