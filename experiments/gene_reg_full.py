import csv

import pandas
import time

from experiments.paths import Paths
from latnet.latnet import Latnet
from expr_util import ExprUtil
from latnet.expr_logger import Logger
from latnet.util import check_dir_exists
import numpy as np


class GeneRegulatoryFull:
    """
    Runs experiment related to gene regulatory network using full Yeast genome.
    """

    @staticmethod
    def gene_regulatory_network_Spellman(output_folder):

        path = Paths.RESULTS + output_folder
        check_dir_exists(path)
        Logger.init_logger(path, "run")
        data = pandas.read_csv(Paths.DATA + 'Spellman/gene6178_Spellman.csv')
        Y = []
        for i in range(data.shape[0]):
            Y.append(data.ix[i,1:].values[:, np.newaxis])

        t = [np.array(range(0, Y[0].shape[0]))[:, np.newaxis]]

        start_time = time.time()
        norm_t = (t[0] - np.mean(t[0])) / np.double(np.std(t[0]))

        elbo_, \
        sigma2_n_, sigma2_g_, \
        mu, sigma2_, \
        alpha_, \
        lengthscale, variance = Latnet.optimize(norm_t, np.hstack(Y),
                                      ['var'],
                                      1, {'var': 150,
                                      'hyp': 1000},
                                      Logger.logger,
                                      init_sigma2_n=1.01E-01,
                                      init_sigma2_g=9.44E-04,
                                      init_lengthscle=3.31E-01,
                                      init_variance=1.66E-01,
                                      lambda_prior=0.5,
                                      lambda_postetior=2. / 3.,
                                      var_lr=0.01,
                                      hyp_lr=0.0001,
                                      n_samples=2,
                                      log_every=1,
                                      callback=ExprUtil.write_to_file_callback(path)
                                      )

        end_time = time.time()
        ExprUtil.write_to_file_callback(path)(alpha_, mu, sigma2_, sigma2_n_, sigma2_g_, lengthscale, variance)

        csvdata = [start_time, end_time, end_time - start_time]
        with open(path + '/timing.csv', "w") as csvFile:
            Fileout = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_ALL)
            Fileout.writerow(csvdata)
        for l in Logger.logger.handlers:
            l.close()


if __name__ == '__main__':
    GeneRegulatoryFull.gene_regulatory_network_Spellman('Spellman_full')

