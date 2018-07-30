import numpy as np
from latnet.expr_logger import Logger

import re


def version_comp(version1, version2):
    def normalize(v):
        return [int(x) for x in re.sub(r'(\.0+)*$','', v).split(".")]
    return cmp(normalize(version1), normalize(version2))


class ExprUtil:

    @staticmethod
    def write_to_file(alpha, mu, sigma2, hyp, path):
        """
        Writes parameters (variational and hyper-parameters) to CSV files.

        Args:
            alpha: numpy array of size N x N.
            mu: numpy array of size N x N.
            sigma2: numpy array of size N x N.
            hyp: numpy array.
            path: string. Path to directory in which result files will be saved.
        """

        Logger.logger.debug("writing results to the file")
        np.savetxt(path + '/mu' + '.csv', mu, delimiter=',', comments='')
        np.savetxt(path + '/sigma2' + '.csv', sigma2, delimiter=',', comments='')
        np.savetxt(path + '/alpha' + '.csv', alpha, delimiter=',', comments='')
        np.savetxt(path + '/p' + '.csv', alpha / (1.0 + alpha), delimiter=',', comments='')
        np.savetxt(path + '/hyp' + '.csv', hyp, delimiter=',', comments='')
        Logger.logger.debug("finished writing results to the file")


    @staticmethod
    def write_to_file_callback(path):
        def toFile(alpha_, mu, sigma2_, sigma2_n_, sigma2_g_, lengthscale, variance):
            ExprUtil.write_to_file(alpha_, mu, sigma2_, np.array([sigma2_n_, sigma2_g_, lengthscale, variance]), path)
        return toFile


