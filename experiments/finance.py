import csv
import pandas
import time
from experiments.paths import Paths
from expr_util import ExprUtil
from latnet.latnet import Latnet
from latnet.expr_logger import Logger
from latnet.util import check_dir_exists
import numpy as np


class Finance:
    """
    Runs experiment related to property prices in Sydney (trust me - it is expensive :-) )
    """

    @staticmethod
    def finance():

        output_folder  = '/finance/'
        path = Paths.RESULTS + output_folder
        check_dir_exists(path)
        Logger.init_logger(path, "run", logger_name=output_folder)

        data = pandas.read_csv(Paths.DATA + 'finance/Returns-2003-2009+Comments.csv', header=0, index_col=0)

        data = np.array(data).T
        t = [np.array(range(0, data.shape[0]))[:, np.newaxis]]
        Logger.logger.debug('size of Y is: %s' % str(data.shape))

        start_time = time.time()
        norm_t = (t[0] - np.mean(t[0])) / np.double(np.std(t[0]))
        elbo_, \
        sigma2_n_, sigma2_g_, \
        mu, sigma2_, \
        alpha_, \
        lengthscale, variance = Latnet.optimize(norm_t, data,
                                      ['var', 'hyp'],
                                      5,
                                      {'var': 500, 'hyp': 10000},
                                      Logger.logger,
                                      init_sigma2_n=0.06,
                                      init_sigma2_g=1e-4,
                                      init_lengthscle=2.,
                                      init_variance=0.6,
                                      lambda_prior=0.5,
                                      lambda_postetior=2. / 3.,
                                      var_lr=0.01,
                                      hyp_lr=0.0001,
                                      n_samples=10
                                      )

        end_time = time.time()
        ExprUtil.write_to_file(alpha_, mu, sigma2_, np.array([sigma2_n_, sigma2_g_, lengthscale, variance]), path)

        csvdata = [start_time, end_time, end_time - start_time]
        with open(path + '/timing.csv', "w") as csvFile:
            Fileout = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_ALL)
            Fileout.writerow(csvdata)
        for l in Logger.logger.handlers:
            l.close()


if __name__ == '__main__':
    Finance.finance()
