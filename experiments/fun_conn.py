import csv
import sys
from multiprocessing import Pool

import pandas
import time

from experiments.paths import Paths
from latnet.latnet import Latnet
from expr_util import ExprUtil
from latnet.expr_logger import Logger
from latnet.util import check_dir_exists
import numpy as np


class FuncConn:
    """
    This class runs experiments related to functional connectivity data. Data is from:
        Smith SM, Miller KL, Salimi-Khorshidi G, et al. Network modelling methods for FMRI. Neuroimage. 2011;54(2):875-891.
        Data was downloaded from here: http://www.fmrib.ox.ac.uk/datasets/netsim/

    """
    def __init__(self):
        pass

    @staticmethod
    def functional_connectivity_group(output_folder, input_file, Ti):
        """
        Args:
            output_folder: the folder in which the results of running LATNET will be saved in.
            input_file: the name of input file that contains observations from nodes.
            Ti: number of observations to use for running the model (from 1...Ti)

        Returns: None
        """

        output_folder = output_folder + str(Ti) + '/'
        data = pandas.read_csv(Paths.DATA + input_file, header=None)

        Y = []
        l = data.values.shape[0]/50
        for s in range(50):
            """s: subject number. Each file contains data fro 50 subjects."""

            Y.append([])
            for i in range(data.shape[1]):
                Y[s].append(data.ix[:, i].values[s*l:(s+1)*l, np.newaxis][0:Ti,:])
            mean_ = np.hstack(Y[s]).mean()
            std_ = np.hstack(Y[s]).std()

            # for standardizing the inputs
            for i in range(data.shape[1]):
                Y[s][i] = (Y[s][i] - mean_) / std_

        id = "latnet"
        for s in range(len(Y)):
            # running LATNET algorithm on data of each subject
            FuncConn.functional_connectivity_sim(Y[s], output_folder + str(id) + '_' + str(s))

    @staticmethod
    def functional_connectivity_sim(Y, folder_name):

        path = Paths.RESULTS + folder_name
        check_dir_exists(path)
        Logger.init_logger(path, "run", folder_name)

        t = [np.array(range(0, Y[0].shape[0]))[:, np.newaxis]]

        start_time = time.time()
        norm_t = (t[0] - np.mean(t[0])) / np.double(np.std(t[0]))
        init_lenthscale = 1. / np.sqrt(norm_t.shape[0])

        lambda_prior = 1.
        lambda_postetior = .15

        elbo_, \
        sigma2_n_, sigma2_g_, \
        mu, sigma2_, \
        alpha_, \
        lengthscale, variance = Latnet.optimize(norm_t, np.hstack(Y),
                                      ['var', 'hyp'],
                                      7, {'var': 2000, 'hyp': 2000},
                                      Logger.logger,
                                      init_sigma2_n=0.31,
                                      init_sigma2_g=1e-4,
                                      init_lengthscle=init_lenthscale,
                                      init_variance=0.50,
                                      lambda_prior=lambda_prior,
                                      lambda_postetior=lambda_postetior,
                                      var_lr=0.01,
                                      hyp_lr=0.001
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

    configs = []

    for sims in ['sim1', 'sim2', 'sim3']:
        """sims: which simulation in the dataset """
        for Ti in [50, 100, 200]:
            """Ti: number of observations"""
            configs.append({'sims': sims, 'Ti':Ti})


    def run_lat(i):
        sims = configs[i]['sims']
        Ti = configs[i]['Ti']
        FuncConn.functional_connectivity_group('fmri/fmri_' + sims + '_LatNet/',
                                               'fmri_sim/ts_' + sims + '.csv', Ti=Ti)

    """ for multi-processing purposes """
    if len(sys.argv) == 2:
        n_proc = int(sys.argv[1])
    elif len(sys.argv) == 1:
        n_proc = 1
    else:
        raise Exception('invalid argument')

    p = Pool(n_proc)
    p.map(run_lat, range(len(configs)))

    p.close()  # no more tasks
    p.join()  # wrap up current tasks

