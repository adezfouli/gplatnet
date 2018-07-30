import csv
import pandas
import time
from experiments.paths import Paths
from expr_util import ExprUtil
from latnet.latnet import Latnet
from latnet.expr_logger import Logger
from latnet.util import check_dir_exists
import numpy as np


class HousePrice:
    """
    Runs experiment related to property prices in Sydney (trust me - it is expensive :-) )
    """

    @staticmethod
    def NSW_house_price(input_file, output_folder, from_, to_):

        path = Paths.RESULTS + output_folder
        check_dir_exists(path)
        Logger.init_logger(path, "run", logger_name=output_folder)

        data = pandas.read_csv(Paths.DATA + input_file)

        data = data.T
        Y = []
        for i in range(0, data.shape[1]):
            Y.append((data.ix[:, i].values[(4 * (from_ - 1995) + 1):(4 * (to_ - 1999) + 21), np.newaxis]))

        mean_ = np.hstack(Y).mean()
        std_ = np.hstack(Y).std()
        for i in range(0, data.shape[1]):
            Y[i] = (Y[i] - mean_) / std_

        t = [np.array(range(0, Y[0].shape[0]))[:, np.newaxis]]
        Logger.logger.debug('size of Y is: %d' % len(Y))

        start_time = time.time()
        norm_t = (t[0] - np.mean(t[0])) / np.double(np.std(t[0]))
        elbo_, \
        sigma2_n_, sigma2_g_, \
        mu, sigma2_, \
        alpha_, \
        lengthscale, variance = Latnet.optimize(norm_t, np.hstack(Y),
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
                                      hyp_lr=0.0001
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
    window_size = 4
    window_slide = 1
    start_pose = 1995
    end_pose = start_pose + window_size
    while end_pose <= 2014:
        file_name = str(start_pose) + '_' + str(end_pose)
        HousePrice.NSW_house_price('NSW_house/data.csv', 'NSW_house/' + file_name + '/', start_pose, end_pose)
        start_pose += window_slide
        end_pose = start_pose + window_size
